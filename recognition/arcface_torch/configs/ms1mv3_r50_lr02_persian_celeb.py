from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r50"
config.resume = False
config.output = "results/run1"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 6
config.lr = 0.2
config.verbose = 50
config.dali = False

config.frequent = 10
config.rec = "data/persian_celeb_"
config.num_classes = 10
config.num_image = 40
config.num_epoch = 40
config.warmup_epoch = 2
config.val_targets = ['pers']
