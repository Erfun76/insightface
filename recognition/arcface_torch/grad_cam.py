import argparse

import cv2
import numpy as np
import torch
from numpy.linalg import norm as l2norm

from backbones import get_model
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class SimilarityToConceptTarget:
    def __init__(self, features):
        # features = torch.nn.functional.normalize(features, p=2.0)
        self.features = features[0]
    
    def __call__(self, model_output):
        # model_output = torch.nn.functional.normalize(model_output, p=2.0)
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)

# @torch.no_grad()
def inference(weight, name, img, img2):
    device = torch.device('cpu')
    if img is None:
        print("None")
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))
        img2 = cv2.imread(img2)
        img2 = cv2.resize(img2, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_img =np.float32(img) / 255
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = np.transpose(img2, (2, 0, 1))
    img2 = torch.from_numpy(img2).unsqueeze(0).float()
    img2.div_(255).sub_(0.5).div_(0.5)


    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight, map_location=device))
    net.eval()
    # with torch.no_grad():
    feat2 = net(img2)
    target_layers = [net.layer4[-1]]
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
    targets = [SimilarityToConceptTarget(feat2)]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # grayscale_cam = cam(input_tensor=img, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    # grayscale_cam = grayscale_cam[0, :]
    # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("output.png", grayscale_cam)
    # Where is the car in the image
    with GradCAM(model=net,
             target_layers=target_layers,
             use_cuda=False) as cam:
        car_grayscale_cam = cam(input_tensor=img, targets=targets, eigen_smooth=True, aug_smooth=True)[0, :]

    car_cam_image = show_cam_on_image(rgb_img, car_grayscale_cam, use_rgb=True)
    car_cam_image = cv2.cvtColor(car_cam_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("results/grad_cam_output.png", car_cam_image)
    # return feat.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()

    feat1 = inference(args.weight, args.network, './data/bato_data/persian_celeb_112x112/oveisi/oveisi_135.jpg', './data/bato_data/persian_celeb_112x112/oveisi/oveisi_478.jpg')