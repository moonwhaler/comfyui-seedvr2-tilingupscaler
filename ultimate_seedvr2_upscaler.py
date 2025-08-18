import torch
import numpy as np
from PIL import Image
import nodes

def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, 'RGB')
    return image

def pil_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class UltimateSeedVR2Upscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ([
                    "seedvr2_ema_3b_fp16.safetensors", 
                    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                    "seedvr2_ema_7b_fp16.safetensors",
                    "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
                    "seedvr2_ema_7b_sharp_fp16.safetensors",
                    "seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors"
                ], {
                    "default": "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
                }),
                "seed": ("INT", {"default": 100, "min": 0, "max": 2**32 - 1, "step": 1}),
                "new_resolution": ("INT", {"default": 1072, "min": 16, "max": 4320, "step": 16}),
                "preserve_vram": ("BOOLEAN", {"default": False}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8}),
                "tile_upscale_resolution": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "block_swap_config": ("block_swap_config",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, seed, new_resolution, preserve_vram, tile_width, tile_height, mask_blur, tile_padding, tile_upscale_resolution, block_swap_config=None):
        # Find the SeedVR2 node class dynamically
        seedvr2_class = None
        for node_class in nodes.NODE_CLASS_MAPPINGS.values():
            if node_class.__name__ == "SeedVR2":
                seedvr2_class = node_class
                break
        
        if not seedvr2_class:
            raise RuntimeError("Could not find SeedVR2 node. Make sure it is installed correctly.")

        seedvr2_instance = seedvr2_class()

        pil_image = tensor_to_pil(image)

        upscale_factor = new_resolution / max(pil_image.width, pil_image.height)
        output_width = int(pil_image.width * upscale_factor)
        output_height = int(pil_image.height * upscale_factor)

        tiles = self._tile_image(pil_image, tile_width, tile_height, tile_padding)
        
        upscaled_tiles = []
        for tile_info in tiles:
            tile_tensor = pil_to_tensor(tile_info["tile"])

            # Stage 1: AI upscale to the fixed tile resolution
            upscaled_tile_tuple = seedvr2_instance.execute(
                images=tile_tensor,
                model=model,
                seed=seed,
                new_resolution=tile_upscale_resolution,
                batch_size=1,
                preserve_vram=preserve_vram,
                block_swap_config=block_swap_config
            )
            
            ai_upscaled_tile = tensor_to_pil(upscaled_tile_tuple[0])

            # Stage 2: Resize the AI upscaled tile to its final target size
            target_tile_width = int(tile_info["tile"].width * upscale_factor)
            target_tile_height = int(tile_info["tile"].height * upscale_factor)
            
            resized_tile = ai_upscaled_tile.resize((target_tile_width, target_tile_height), Image.LANCZOS)

            upscaled_tiles.append({
                "tile": resized_tile,
                "position": (int(tile_info["position"][0] * upscale_factor), int(tile_info["position"][1] * upscale_factor))
            })

        output_image = self._stitch_tiles(upscaled_tiles, output_width, output_height, int(tile_width * upscale_factor), int(tile_height * upscale_factor), int(tile_padding * upscale_factor), mask_blur)

        return (pil_to_tensor(output_image),)

    def _tile_image(self, image, tile_width, tile_height, padding):
        width, height = image.size
        tiles = []
        for y in range(0, height, tile_height):
            for x in range(0, width, tile_width):
                box = (x, y, x + tile_width, y + tile_height)
                padded_box = (
                    max(0, box[0] - padding),
                    max(0, box[1] - padding),
                    min(width, box[2] + padding),
                    min(height, box[3] + padding),
                )
                tile = image.crop(padded_box)
                tiles.append({
                    "tile": tile,
                    "position": (x, y)
                })
        return tiles

    def _stitch_tiles(self, tiles, width, height, tile_width, tile_height, padding, mask_blur):
        output_image = Image.new('RGB', (width, height))
        for tile_info in tiles:
            tile = tile_info["tile"]
            x, y = tile_info["position"]

            # Crop the padding from the tile
            cropped_tile = tile.crop((padding, padding, tile.width - padding, tile.height - padding))

            # Create a feathered mask for blending
            mask = self._create_feathered_mask(cropped_tile.width, cropped_tile.height, mask_blur)
            
            output_image.paste(cropped_tile, (x, y), mask)
            
        return output_image

    def _create_feathered_mask(self, width, height, blur_radius):
        mask = Image.new('L', (width, height), 255)
        
        # Create a feathered edge on all four sides
        for i in range(blur_radius):
            alpha = int(255 * (i / blur_radius))
            
            # Top and bottom edges
            for x in range(width):
                mask.putpixel((x, i), alpha)
                mask.putpixel((x, height - 1 - i), alpha)
                
            # Left and right edges
            for y in range(height):
                mask.putpixel((i, y), alpha)
                mask.putpixel((width - 1 - i, y), alpha)
                
        return mask

NODE_CLASS_MAPPINGS = {
    "UltimateSeedVR2Upscaler": UltimateSeedVR2Upscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSeedVR2Upscaler": "Ultimate SeedVR2 Upscaler"
}
