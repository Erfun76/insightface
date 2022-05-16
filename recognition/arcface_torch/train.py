import argparse
import logging
import os
# from turtle import back

import torch
from torch import distributed
from torch.utils.tensorboard import SummaryWriter

# from torchsummary import summary
from torch import nn
from backbones import get_model
from dataset import get_dataloader
from torch.utils.data import DataLoader
from lr_scheduler import PolyScheduler
from losses import CosFace, ArcFace
from partial_fc import PartialFC
from utils.utils_callbacks import CallBackLoggingResume, CallBackVerification, CallBackModelCheckpointResume
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging


try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    torch.cuda.set_device(args.local_rank)
    cfg = get_config(args.config)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)
    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    train_loader = get_dataloader(
        cfg.rec, local_rank=args.local_rank, batch_size=cfg.batch_size, dali=cfg.dali)

    if cfg.loss == "arcface":
        margin_loss = ArcFace()
    elif cfg.loss == "cosface":
        margin_loss = CosFace()
    else:
        raise

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size
    ).cuda()
    # backbone._fc =  nn.Sequential(nn.Linear(1280, 512), nn.BatchNorm1d(512, eps=1e-05))

    # summary(backbone, (3,112,112))

    module_partial_fc = PartialFC(
        margin_loss,
        cfg.embedding_size, 
        cfg.num_classes, 
        cfg.sample_rate, 
        cfg.fp16
    )

    total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // total_batch_size * cfg.num_epoch
    start_epoch = 0
    global_step = 0
    # changed
    if args.pretrain:
            backbone.load_state_dict(torch.load("results/backbone_r100.pth", map_location=torch.device(args.local_rank)))

    if cfg.resume:
        # try:   
            backbone_pth = os.path.join(cfg.output, "savedckpt.pth")
            savedckpt = torch.load(backbone_pth, map_location=torch.device(args.local_rank))
            start_epoch = savedckpt['epoch'] + 1
            global_step = int(cfg.num_image/cfg.batch_size) * (savedckpt['epoch']+1) + 1 
            backbone.load_state_dict(savedckpt['backbone'].module.state_dict())
            head_pth = os.path.join(cfg.output, "softmax_fc_gpu_0.pt")
            module_partial_fc.load_state_dict(torch.load(head_pth, map_location=torch.device(args.local_rank)))

            if rank == 0:
                logging.info("backbone resume successfully!")
        # except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        #     if rank == 0:
        #         logging.info("resume fail, backbone init successfully!")
    else:
        start_epoch = 0
        global_step = 0
    
    

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank])
    backbone.train()

    module_partial_fc.train().cuda()

    # TODO the params of partial fc must be last in the params list
    opt = torch.optim.SGD(
        params=[
            {"params": backbone.parameters(), },
            {"params": module_partial_fc.parameters(), },
        ],
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay
    )
    
    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step
    )

    if cfg.resume:
        # try: 
            savedckpt = torch.load(backbone_pth, map_location=torch.device(args.local_rank))
            opt.load_state_dict(savedckpt['optimizer'])
            lr_scheduler.load_state_dict(savedckpt['scheduler'])
        # except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        #     if rank == 0:
        #         logging.info("resume fail, backbone init successfully!")

    
    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )
    # callback_logging = CallBackLogging(
    #     frequent=cfg.frequent,
    #     total_step=cfg.total_step,
    #     batch_size=cfg.batch_size,
    #     writer=summary_writer
    # )
    # changed
    callback_logging = CallBackLoggingResume(cfg.frequent, rank, cfg.total_step, global_step, cfg.batch_size, world_size, summary_writer)
    callback_checkpoint = CallBackModelCheckpointResume(rank, cfg.output)

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels, opt)

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)


                if global_step % cfg.verbose == 0 and global_step > 5:
                    path_module = os.path.join(cfg.output, "best_model.pt")
                    callback_verification(global_step, backbone, path_module)

        if epoch % cfg.frequent == 0:
            callback_checkpoint(global_step, epoch, backbone, module_partial_fc, opt, lr_scheduler)

        # path_pfc = os.path.join(cfg.output, "softmax_fc_gpu_{}.pt".format(rank))
        # torch.save(module_partial_fc.state_dict(), path_pfc)
        # if rank == 0:
        #     path_module = os.path.join(cfg.output, "model.pt")
        #     torch.save(backbone.module.state_dict(), path_module)
        
        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    parser.add_argument("--pretrain", type=bool, default=False)
    main(parser.parse_args())
