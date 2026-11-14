"""Ai2's MolmoAct2 fine-tuning image augmentation, as a transform Studio's LeRobotDataModule can load.

The recipe (allenai/molmoact2, mirrored in LeRobot's MolmoAct2 docs) applies, in order: a 95%-per-side crop
(area 0.9025) resized back to the input size, a random rotation within ±5°, colour jitter (brightness 0.2,
contrast 0.8–1.2, saturation 0.8–1.2, hue 0.05), and, in Ai2's original stack, a Gaussian blur with
probability 0.2. LeRobot's port can't express the blur; this one can.

LeRobot calls the transform once per camera image (C, H, W float in [0, 1]), so each camera gets its own
random parameters and its own size (wrist 480x640, cropped top 480x560). Shapes are preserved.
"""

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


class Ai2Augment(torch.nn.Module):
    def __init__(self, crop_side: float = 0.95, degrees: float = 5.0, brightness: float = 0.2,
                 contrast: tuple[float, float] = (0.8, 1.2), saturation: tuple[float, float] = (0.8, 1.2),
                 hue: float = 0.05, blur_p: float = 0.2, blur_kernel: int = 5, blur_sigma=(0.1, 2.0)):
        super().__init__()
        if not 0 < crop_side <= 1:
            raise ValueError("crop_side must be in (0, 1]")
        self.crop_side = crop_side
        self.rotate = v2.RandomRotation(degrees, interpolation=v2.InterpolationMode.BILINEAR)
        self.jitter = v2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.blur = v2.RandomApply([v2.GaussianBlur(blur_kernel, sigma=blur_sigma)], p=blur_p)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        ch, cw = max(1, round(h * self.crop_side)), max(1, round(w * self.crop_side))
        top = int(torch.randint(0, h - ch + 1, ()).item())
        left = int(torch.randint(0, w - cw + 1, ()).item())
        image = F.resized_crop(image, top, left, ch, cw, [h, w], antialias=True)
        image = self.jitter(self.rotate(image))
        return self.blur(image).clamp(0.0, 1.0) if image.is_floating_point() else self.blur(image)
