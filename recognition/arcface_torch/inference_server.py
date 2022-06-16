import argparse

import cv2
import numpy as np
import torch
from numpy.linalg import norm as l2norm
from sklearn import preprocessing

from backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    device = torch.device("cpu")
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    net.to(device)
    img = img.to(device)
    for _ in range(10):
        _ = net(img)
    start.record()
    feat = net(img).cpu().numpy()
    end.record()
    torch.cuda.synchronize()

    print(start.elapsed_time(end))
    return feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')

    args = parser.parse_args()
    feat1 = inference(args.weight, args.network, '/home/erfun/Documents/erfan/mohaymen/insightface/recognition/arcface_torch/data/bato_data/persian_celeb_112x112/attila/attila_63.jpg')
    feat2 = inference(args.weight, args.network, '/home/erfun/Documents/erfan/mohaymen/insightface/recognition/arcface_torch/data/bato_data/persian_celeb_112x112/attila/attila_267.jpg')
    norm_feat1 = feat1[0]/l2norm(feat1[0])
    norm_feat2 = feat2[0]/l2norm(feat2[0])
    print(np.dot(norm_feat1,norm_feat2))
