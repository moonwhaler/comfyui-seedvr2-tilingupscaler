"""Image utility functions for tensor/PIL conversions."""

import torch
import numpy as np
from PIL import Image


def tensor_to_pil(tensor):
    """Convert a tensor to PIL Image."""
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, 'RGB')
    return image


def pil_to_tensor(image):
    """Convert a PIL Image to tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
