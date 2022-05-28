import sys

sys.path.append('./MiDaS')
import torch
import cv2

from torchvision.transforms import Compose
from models.midas_net import MidasNet
from models.transforms import Resize, NormalizeImage, PrepareForNet

from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

model = transform = device = depth = None


def load_model():
    global model, transform, device
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model = MidasNet('./MiDaS/model.pt', non_negative=True)
    transform = Compose(
        [
            Resize(
                384 * 2,
                384 * 2,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
    model.to(device)
    model.eval()


def predict_depth(image):
    global model, transform, device, depth
    image = image.astype('float32') / 255.
    img_input = transform({"image": image})["image"]
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        depth_min = prediction.min()
        depth_max = prediction.max()
        out = 255 * (prediction - depth_min) / (depth_max - depth_min)
        depth = out.astype('uint8')
        return depth


layers = image = None


def get_layers(depth_img=None, num_layers=2, return_image=False, return_layers=True):
    global depth, layers, image
    if depth_img is None:
        image = depth
    else:
        image = depth_img
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    original_shape = image.copy().shape
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=num_layers)
    clt.fit(image)

    if return_image:
        result = [np.uint8(clt.cluster_centers_[lab]) for lab in clt.labels_]
        result = np.reshape(result, original_shape)
        return result

    if return_layers:
        ordered_labels_idx = np.argsort(np.array(clt.cluster_centers_)[:, 0])
        result = np.reshape(np.array(clt.labels_), original_shape[:2])
        layers = [np.uint8(result == layer) * 255 for layer in ordered_labels_idx]
        return layers


def create_animation(img, num_frames=50, delta=0.01):
    global layers
    height, width = layers[0].shape[:2]

    front = cv2.cvtColor(layers[-1], cv2.COLOR_GRAY2BGR)

    gif_array = []
    aux_image = img.copy()
    for frame in range(num_frames):
        aux_image = cv2.resize(aux_image, None, fx=1+frame*delta/2, fy=1+frame*delta/2, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(front.copy(), None, fx=1+frame*delta, fy=1+frame*delta, interpolation=cv2.INTER_CUBIC)

        new_width = aux_image.shape[1]
        new_height = aux_image.shape[0]

        more_x = aux_image.shape[1] - width
        more_y = aux_image.shape[0] - height

        aux_image = aux_image[more_y // 2: new_height - more_y // 2, more_x // 2: new_width - more_x // 2]
        mask = mask[more_y // 2: new_height - more_y // 2, more_x // 2: new_width - more_x // 2]

        result = np.where(mask == 255, aux_image, img.copy()).astype(np.uint8)

        gif_array.append(result)

    return gif_array

