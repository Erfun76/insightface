from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r100"
config.resume = False
config.output = "results/run13"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 64
config.lr = 0.0002
config.verbose = 4500
config.dali = False

config.frequent = 10
config.rec = "/app/insightface_preprocessing/dataset_112x112_cleaned"
config.num_classes = 2056
config.num_image = 131057
config.num_epoch = 30
config.warmup_epoch = 2
config.val_targets = ['lfw', 'valid']
