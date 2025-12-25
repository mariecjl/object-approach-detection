import torch
import cv2
import numpy as np

#loading midas
def load_midas():
    model_type = "MiDaS_small" #midas small for faster performance when code runs
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    #transforms
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform

    #cuda gpu/cpu allocation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)

    return midas, transform, device


@torch.no_grad()
def estimate_depth(frame, midas, transform, device):
    h, w, _ = frame.shape

    #image preprocessing
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    #depth calculation
    depth = midas(input_batch)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False
    ).squeeze()

    #np array conversion and smoothing depth map
    depth = depth.cpu().numpy()
    depth = cv2.GaussianBlur(depth, (7, 7), 0)

    return depth
