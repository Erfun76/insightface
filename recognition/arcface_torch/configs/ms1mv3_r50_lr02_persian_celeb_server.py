from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r50"
config.resume = False
config.output = "results/run5"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.02
config.verbose = 5000
config.dali = False

config.frequent = 10
config.rec = "/app/dataset_112x112_cleaned"
config.num_classes = 2056
config.num_image = 131057
config.num_epoch = 20
config.warmup_epoch = 2
config.val_targets = ['lfw', 'valid']
