import torchvision.transforms as T
from src.core import TRANSFORMS

@TRANSFORMS.register("Resize")
class Resize:
    def __init__(self, size):
        self.t = T.Resize(size)
    def __call__(self, img):
        return self.t(img)

@TRANSFORMS.register("Normalize")
class Normalize:
    def __init__(self, mean, std):
        self.t = T.Normalize(mean=mean, std=std)
    def __call__(self, img):
        return self.t(img)

@TRANSFORMS.register("ToTensor")
class ToTensor:
    def __init__(self):
        self.t = T.ToTensor()
    def __call__(self, img):
        return self.t(img)