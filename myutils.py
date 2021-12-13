import cv2
import torch
import numpy as np

# Values from lrp-tutorial and zennit
ILSVRC2012_MEAN = [0.485, 0.456, 0.406]
ILSVRC2012_STD = [0.229, 0.224, 0.225]


# Source: https://github.com/chr5tphr/zennit/blob/cc9ac0f3016e1b842f2c60e8986c794b2ae7096e/share/example/feed_forward.py#L32-L38
class BatchNormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)[None, :, None, None]
        self.std = torch.tensor(std)[None, :, None, None]

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class ILSVRC2012_BatchNormalize(BatchNormalize):
    def __init__(self):
        super().__init__(mean=ILSVRC2012_MEAN, std=ILSVRC2012_STD)


# Source: https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/38831a1ce9eeb9268e9bb03561d8b9f4828d7e3d/tutorial.ipynb
def load_normalized_img(path):
    # Returns a numpy array in BGR color space, not RGB
    img = cv2.imread(path)

    # Convert from BGR to RGB color space
    img = img[..., ::-1]

    # img.shape is (224, 224, 3), where 3 corresponds to RGB channels
    # Divide by 255 (max. RGB value) to normalize pixel values to [0,1]
    return img/255.0

# Custom function
# Inspired by https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/38831a1ce9eeb9268e9bb03561d8b9f4828d7e3d/tutorial.ipynb and
# zennit


def img_to_tensor(img):
    # reshape converts row vectors to column vectors
    # {mean and std have shape torch.Size([1, 3, 1, 1, 1])

    # Mean and std are calculated from the dataset ImageNet
    # https://github.com/Cadene/pretrained-models.pytorch/blob/8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0/pretrainedmodels/models/torchvision_models.py#L64-L65
    # mean = torch.Tensor(ILSVRC2012_MEAN).reshape(1, -1, 1, 1)
    # std = torch.Tensor(ILSVRC2012_STD).reshape(1, -1, 1, 1)

    # X has shape (1, 3, 224, 224)
    # Normalize X by subtracting mean and dividing by standard deviation
    X = ILSVRC2012_BatchNormalize()(torch.FloatTensor(img[np.newaxis].transpose(
        [0, 3, 1, 2])*1))
    return X
