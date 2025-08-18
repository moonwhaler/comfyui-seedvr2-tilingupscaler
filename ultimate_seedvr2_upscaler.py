import torch
import numpy as np
from PIL import Image
import nodes

class Progress:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0

    def update(self):
        self.current_step += 1
        print(f"Processing step {self.current_step}/{self.total_steps}...")

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
                "mask_blur": ("INT", {"default": 16, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8}),
                "tile_upscale_resolution": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "tiling_strategy": (["Chess", "Linear"],),
                "seam_fix_mode": (["Disabled", "Simple", "Advanced"],),
                "seam_fix_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 8192, "step": 8}),
                "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": 8192, "step": 8}),
            },
            "optional": {
                "block_swap_config": ("block_swap_config",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, seed, new_resolution, preserve_vram, tile_width, tile_height, mask_blur, tile_padding, tile_upscale_resolution, tiling_strategy, seam_fix_mode, seam_fix_denoise, seam_fix_width, seam_fix_mask_blur, seam_fix_padding, block_swap_config=None):
        # Setup
        seedvr2_instance = self._get_seedvr2_instance()
        pil_image = tensor_to_pil(image)
        upscale_factor = new_resolution / max(pil_image.width, pil_image.height)
        output_width = int(pil_image.width * upscale_factor)
        output_height = int(pil_image.height * upscale_factor)

        # Progress tracking
        total_steps = self._calculate_total_steps(pil_image, tile_width, tile_height, seam_fix_mode, seam_fix_width)
        progress = Progress(total_steps)

        # Store original image for base creation
        self._original_image = pil_image
        
        # Main upscale pass
        main_tiles = self._generate_tiles(pil_image, tile_width, tile_height, tile_padding, tiling_strategy)
        output_image = self._process_and_stitch(main_tiles, output_width, output_height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, mask_blur, progress)

        # Seam fix pass
        if seam_fix_mode != "Disabled":
            output_image = self._seam_fix(output_image, pil_image, tile_width, tile_height, seam_fix_width, seam_fix_padding, seam_fix_mask_blur, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, seam_fix_mode, progress)

        return (pil_to_tensor(output_image),)

    def _get_seedvr2_instance(self):
        for node_class in nodes.NODE_CLASS_MAPPINGS.values():
            if node_class.__name__ == "SeedVR2":
                return node_class()
        raise RuntimeError("Could not find SeedVR2 node. Make sure it is installed correctly.")

    def _calculate_total_steps(self, image, tile_width, tile_height, seam_fix_mode, seam_fix_width):
        main_tiles = self._generate_tiles(image, tile_width, tile_height, 0, "Linear")
        total = len(main_tiles)
        if seam_fix_mode != "Disabled":
            seam_tiles = self._generate_seam_tiles(image, tile_width, tile_height, seam_fix_width, 0)
            total += len(seam_tiles)
            if seam_fix_mode == "Advanced":
                intersection_tiles = self._generate_intersection_tiles(image, tile_width, tile_height, seam_fix_width, 0)
                total += len(intersection_tiles)
        return total

    def _generate_tiles(self, image, tile_width, tile_height, padding, strategy):
        width, height = image.size
        tiles = []
        
        if strategy == "Linear":
            for y in range(0, height, tile_height):
                for x in range(0, width, tile_width):
                    tiles.append(self._get_tile_info(image, x, y, tile_width, tile_height, padding))
        elif strategy == "Chess":
            for y_idx, y in enumerate(range(0, height, tile_height)):
                for x_idx, x in enumerate(range(0, width, tile_width)):
                    if (x_idx + y_idx) % 2 == 0:
                        tiles.append(self._get_tile_info(image, x, y, tile_width, tile_height, padding))
            for y_idx, y in enumerate(range(0, height, tile_height)):
                for x_idx, x in enumerate(range(0, width, tile_width)):
                    if (x_idx + y_idx) % 2 != 0:
                        tiles.append(self._get_tile_info(image, x, y, tile_width, tile_height, padding))
        return tiles

    def _get_tile_info(self, image, x, y, tile_width, tile_height, padding):
        width, height = image.size
        
        # Calculate actual tile boundaries (may be smaller at edges)
        actual_tile_width = min(tile_width, width - x)
        actual_tile_height = min(tile_height, height - y)
        
        # Calculate padding (only add padding where there are adjacent tiles)
        left_pad = padding if x > 0 else 0
        top_pad = padding if y > 0 else 0
        right_pad = padding if x + tile_width < width else 0
        bottom_pad = padding if y + tile_height < height else 0

        # Create the padded crop box
        padded_box = (
            max(0, x - left_pad),
            max(0, y - top_pad),
            min(width, x + actual_tile_width + right_pad),
            min(height, y + actual_tile_height + bottom_pad),
        )
        
        tile = image.crop(padded_box)
        return {
            "tile": tile,
            "position": (x, y),
            "actual_size": (actual_tile_width, actual_tile_height),
            "padding": (left_pad, top_pad, right_pad, bottom_pad),
        }

    def _process_and_stitch(self, tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, mask_blur, progress, base_image=None):
        # Create base image if not provided
        if base_image is None:
            # For the main pass, create a better base using SeedVR2 at low resolution
            if hasattr(self, '_original_image'):
                # Create base by upscaling original at low resolution for better quality
                base_tensor = pil_to_tensor(self._original_image)
                base_upscaled_tuple = seedvr2_instance.execute(
                    images=base_tensor, 
                    model=model, 
                    seed=seed, 
                    new_resolution=min(512, tile_upscale_resolution//2),  # Lower res for base
                    batch_size=1, 
                    preserve_vram=preserve_vram, 
                    block_swap_config=block_swap_config
                )
                base_upscaled = tensor_to_pil(base_upscaled_tuple[0])
                base_image = base_upscaled.resize((width, height), Image.LANCZOS)
            else:
                base_image = Image.new('RGB', (width, height), (128, 128, 128))
        
        output_image = base_image.copy()

        # Process all tiles
        for tile_info in tiles:
            progress.update()
            
            # Upscale tile
            tile_tensor = pil_to_tensor(tile_info["tile"])
            upscaled_tile_tuple = seedvr2_instance.execute(images=tile_tensor, model=model, seed=seed, new_resolution=tile_upscale_resolution, batch_size=1, preserve_vram=preserve_vram, block_swap_config=block_swap_config)
            ai_upscaled_tile = tensor_to_pil(upscaled_tile_tuple[0])

            # Resize to final target size
            target_tile_width = int(tile_info["tile"].width * upscale_factor)
            target_tile_height = int(tile_info["tile"].height * upscale_factor)
            resized_tile = ai_upscaled_tile.resize((target_tile_width, target_tile_height), Image.LANCZOS)
            
            # Calculate the region this tile covers (without padding)
            paste_x = int(tile_info["position"][0] * upscale_factor)
            paste_y = int(tile_info["position"][1] * upscale_factor)
            final_tile_width = int(tile_info["actual_size"][0] * upscale_factor)
            final_tile_height = int(tile_info["actual_size"][1] * upscale_factor)
            
            # Calculate scaled padding to crop correctly
            left_pad, top_pad, right_pad, bottom_pad = tile_info["padding"]
            scaled_left_pad = int(left_pad * upscale_factor)
            scaled_top_pad = int(top_pad * upscale_factor)
            
            # Crop the upscaled tile to remove padding
            crop_box = (
                scaled_left_pad,
                scaled_top_pad,
                scaled_left_pad + final_tile_width,
                scaled_top_pad + final_tile_height
            )
            cropped_tile = resized_tile.crop(crop_box)
            
            # Create a feathered mask for better blending at tile edges
            tile_mask = self._create_feathered_tile_mask(final_tile_width, final_tile_height, mask_blur, tile_info["padding"])
            
            # Create RGBA version of the tile for compositing
            tile_rgba = Image.new('RGBA', output_image.size, (0, 0, 0, 0))
            tile_rgba.paste(cropped_tile, (paste_x, paste_y))
            
            # Create full-size mask
            full_mask = Image.new('L', output_image.size, 0)
            full_mask.paste(tile_mask, (paste_x, paste_y))
            tile_rgba.putalpha(full_mask)
            
            # Alpha composite onto the output image
            output_rgba = output_image.convert('RGBA')
            output_rgba.alpha_composite(tile_rgba)
            output_image = output_rgba.convert('RGB')

        return output_image

    def _create_feathered_tile_mask(self, width, height, blur_radius, padding_info):
        """Create a smooth distance-based mask for seamless blending"""
        mask = Image.new('L', (width, height), 255)
        
        left_pad, top_pad, right_pad, bottom_pad = padding_info
        
        if blur_radius > 0:
            # Use numpy for more efficient and smoother mask creation
            mask_array = np.full((height, width), 255, dtype=np.uint8)
            
            # Create distance-based feathering for smoother transitions
            for y in range(height):
                for x in range(width):
                    distances = []
                    
                    # Calculate distance to edges that should be feathered
                    if left_pad > 0:
                        distances.append(x)
                    if right_pad > 0:
                        distances.append(width - x - 1)
                    if top_pad > 0:
                        distances.append(y)
                    if bottom_pad > 0:
                        distances.append(height - y - 1)
                    
                    if distances:
                        min_dist = min(distances)
                        if min_dist < blur_radius:
                            # Smooth falloff using cosine function for better blending
                            fade_factor = (1 + np.cos(np.pi * (blur_radius - min_dist) / blur_radius)) / 2
                            mask_array[y, x] = int(255 * fade_factor)
            
            mask = Image.fromarray(mask_array)
                            
        return mask

    def _seam_fix(self, image, original_image, tile_width, tile_height, seam_fix_width, seam_fix_padding, seam_fix_mask_blur, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, seam_fix_mode, progress):
        if seam_fix_mode == "Simple" or seam_fix_mode == "Advanced":
            seam_tiles = self._generate_seam_tiles(original_image, tile_width, tile_height, seam_fix_width, seam_fix_padding)
            image = self._process_and_stitch(seam_tiles, image.width, image.height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, seam_fix_mask_blur, progress, base_image=image)

        if seam_fix_mode == "Advanced":
            intersection_tiles = self._generate_intersection_tiles(original_image, tile_width, tile_height, seam_fix_width, seam_fix_padding)
            image = self._process_and_stitch(intersection_tiles, image.width, image.height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, seam_fix_mask_blur, progress, base_image=image)
            
        return image

    def _generate_seam_tiles(self, image, tile_width, tile_height, seam_width, padding):
        width, height = image.size
        tiles = []
        
        for y in range(tile_height, height, tile_height):
            for x in range(0, width, tile_width):
                tiles.append(self._get_tile_info(image, x, y - seam_width // 2, tile_width, seam_width, padding))
                
        for x in range(tile_width, width, tile_width):
            for y in range(0, height, tile_height):
                tiles.append(self._get_tile_info(image, x - seam_width // 2, y, seam_width, tile_height, padding))
        return tiles

    def _generate_intersection_tiles(self, image, tile_width, tile_height, seam_width, padding):
        width, height = image.size
        tiles = []
        
        for y in range(tile_height, height, tile_height):
            for x in range(tile_width, width, tile_width):
                tiles.append(self._get_tile_info(image, x - seam_width // 2, y - seam_width // 2, seam_width, seam_width, padding))
        return tiles

    def _create_linear_mask(self, width, height, blur_radius):
        mask = np.ones((height, width), dtype=np.float32)
        
        if blur_radius > 0:
            for i in range(blur_radius):
                fade = i / blur_radius
                mask[i, :] *= fade
                mask[-i-1, :] *= fade
                mask[:, i] *= fade
                mask[:, -i-1] *= fade
                
        return Image.fromarray((mask * 255).astype(np.uint8))

NODE_CLASS_MAPPINGS = {
    "UltimateSeedVR2Upscaler": UltimateSeedVR2Upscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSeedVR2Upscaler": "Ultimate SeedVR2 Upscaler"
}
