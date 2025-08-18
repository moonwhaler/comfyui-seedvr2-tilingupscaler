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
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8}),
                "tile_upscale_resolution": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "tiling_strategy": (["Chess", "Linear"],),
                "seam_fix_mode": (["Disabled", "Simple", "Advanced"],),
                "seam_fix_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 8192, "step": 8}),
                "seam_fix_mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
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
        
        # Main upscale pass with detail-preserving stitching
        main_tiles = self._generate_tiles(pil_image, tile_width, tile_height, tile_padding, tiling_strategy)
        output_image = self._process_and_stitch(main_tiles, output_width, output_height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, mask_blur, progress)

        # Seam fix pass - eliminate remaining artifacts with user-controlled blur
        if seam_fix_mode != "Disabled":
            print("Applying seam fix to eliminate remaining artifacts...")
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
            seam_tiles = self._generate_targeted_seam_tiles(image, tile_width, tile_height, seam_fix_width, 0)
            total += len(seam_tiles)
            if seam_fix_mode == "Advanced":
                intersection_tiles = self._generate_targeted_intersection_tiles(image, tile_width, tile_height, seam_fix_width, 0)
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
        # Use different stitching strategies based on blur setting
        if mask_blur == 0:
            print("Using zero-blur mode for maximum detail preservation...")
            return self._process_and_stitch_zero_blur(tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, progress, base_image)
        else:
            return self._process_and_stitch_blended(tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, mask_blur, progress, base_image)

    def _process_and_stitch_zero_blur(self, tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, progress, base_image=None):
        """Zero-blur stitching that preserves maximum detail through precise pixel averaging"""
        # Create base image if not provided
        if base_image is None:
            if hasattr(self, '_original_image'):
                base_tensor = pil_to_tensor(self._original_image)
                base_upscaled_tuple = seedvr2_instance.execute(
                    images=base_tensor, 
                    model=model, 
                    seed=seed, 
                    new_resolution=min(512, tile_upscale_resolution//2),
                    batch_size=1, 
                    preserve_vram=preserve_vram, 
                    block_swap_config=block_swap_config
                )
                base_upscaled = tensor_to_pil(base_upscaled_tuple[0])
                base_image = base_upscaled.resize((width, height), Image.LANCZOS)
            else:
                base_image = Image.new('RGB', (width, height), (128, 128, 128))
        
        # Use numpy arrays for precise pixel control
        output_array = np.array(base_image, dtype=np.float64)
        weight_array = np.zeros((height, width), dtype=np.float64)
        
        # Process all tiles with precise pixel averaging
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
            
            # Calculate positioning
            paste_x = int(tile_info["position"][0] * upscale_factor)
            paste_y = int(tile_info["position"][1] * upscale_factor)
            final_tile_width = int(tile_info["actual_size"][0] * upscale_factor)
            final_tile_height = int(tile_info["actual_size"][1] * upscale_factor)
            
            # Calculate scaled padding
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
            tile_array = np.array(cropped_tile, dtype=np.float64)
            
            # Define the region in the output image
            end_x = min(paste_x + final_tile_width, width)
            end_y = min(paste_y + final_tile_height, height)
            
            # Pixel-perfect weighted averaging for seamless blending
            for y in range(paste_y, end_y):
                for x in range(paste_x, end_x):
                    tile_x = x - paste_x
                    tile_y = y - paste_y
                    
                    if tile_y < tile_array.shape[0] and tile_x < tile_array.shape[1]:
                        current_weight = weight_array[y, x]
                        new_weight = current_weight + 1.0
                        
                        # Weighted average - preserves all detail while eliminating seams
                        if current_weight > 0:
                            output_array[y, x] = (output_array[y, x] * current_weight + tile_array[tile_y, tile_x]) / new_weight
                        else:
                            output_array[y, x] = tile_array[tile_y, tile_x]
                        
                        weight_array[y, x] = new_weight
        
        # Convert back to PIL Image
        output_array = np.clip(output_array, 0, 255).astype(np.uint8)
        return Image.fromarray(output_array)

    def _process_and_stitch_blended(self, tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, mask_blur, progress, base_image=None):
        """Standard blended stitching with user-controlled blur"""
        # Create base image if not provided
        if base_image is None:
            if hasattr(self, '_original_image'):
                base_tensor = pil_to_tensor(self._original_image)
                base_upscaled_tuple = seedvr2_instance.execute(
                    images=base_tensor, 
                    model=model, 
                    seed=seed, 
                    new_resolution=min(512, tile_upscale_resolution//2),
                    batch_size=1, 
                    preserve_vram=preserve_vram, 
                    block_swap_config=block_swap_config
                )
                base_upscaled = tensor_to_pil(base_upscaled_tuple[0])
                base_image = base_upscaled.resize((width, height), Image.LANCZOS)
            else:
                base_image = Image.new('RGB', (width, height), (128, 128, 128))
        
        output_image = base_image.copy()

        # Process all tiles with controlled blur blending
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
            
            # Create mask with user-specified blur - respects exact setting
            tile_mask = self._create_precise_tile_mask(final_tile_width, final_tile_height, mask_blur, tile_info["padding"])
            
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

    def _create_precise_tile_mask(self, width, height, blur_radius, padding_info):
        """Create smart blending mask - zero blur interior, minimal blur at seams"""
        left_pad, top_pad, right_pad, bottom_pad = padding_info
        mask_array = np.full((height, width), 255, dtype=np.uint8)
        
        # Smart blend: Use minimal blur only where absolutely needed
        if blur_radius > 0:
            # Effective blur - never more than 3 pixels for seam hiding
            effective_blur = min(blur_radius, 3)
            
            for y in range(height):
                for x in range(width):
                    min_alpha = 255
                    
                    # Apply gradient only at tile boundaries
                    if left_pad > 0 and x < effective_blur:
                        min_alpha = min(min_alpha, int(255 * (x / effective_blur)))
                    
                    if right_pad > 0 and x >= width - effective_blur:
                        min_alpha = min(min_alpha, int(255 * ((width - x - 1) / effective_blur)))
                    
                    if top_pad > 0 and y < effective_blur:
                        min_alpha = min(min_alpha, int(255 * (y / effective_blur)))
                    
                    if bottom_pad > 0 and y >= height - effective_blur:
                        min_alpha = min(min_alpha, int(255 * ((height - y - 1) / effective_blur)))
                    
                    mask_array[y, x] = min_alpha
        
        return Image.fromarray(mask_array)

    def _seam_fix(self, image, original_image, tile_width, tile_height, seam_fix_width, seam_fix_padding, seam_fix_mask_blur, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, seam_fix_mode, progress):
        # Enhanced seam fix for minimal blur scenarios
        if seam_fix_mode == "Simple" or seam_fix_mode == "Advanced":
            print(f"Processing {seam_fix_mode.lower()} seam fix with intelligent seam targeting...")
            seam_tiles = self._generate_targeted_seam_tiles(original_image, tile_width, tile_height, seam_fix_width, seam_fix_padding)
            # Use smart blur for seam fix - minimal blur for detail preservation
            seam_blur = max(seam_fix_mask_blur, 2) if seam_fix_mask_blur > 0 else 2
            image = self._process_and_stitch(seam_tiles, image.width, image.height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, seam_blur, progress, base_image=image)

        if seam_fix_mode == "Advanced":
            print("Processing intersection seam fix with targeted blending...")
            intersection_tiles = self._generate_targeted_intersection_tiles(original_image, tile_width, tile_height, seam_fix_width, seam_fix_padding)
            seam_blur = max(seam_fix_mask_blur, 2) if seam_fix_mask_blur > 0 else 2
            image = self._process_and_stitch(intersection_tiles, image.width, image.height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, seam_blur, progress, base_image=image)
            
        return image

    def _generate_targeted_seam_tiles(self, image, tile_width, tile_height, seam_width, padding):
        """Generate seam tiles with better targeting of actual seam locations"""
        width, height = image.size
        tiles = []
        
        # Narrow seam width for precision - use smaller strips for better targeting
        effective_seam_width = max(seam_width // 2, 32)  # Minimum 32 pixels for context
        
        # Horizontal seams - target exact tile boundaries
        for y in range(tile_height, height, tile_height):
            for x in range(0, width, tile_width):
                # Center seam exactly on tile boundary
                seam_y = y - effective_seam_width // 2
                tiles.append(self._get_tile_info(image, x, seam_y, tile_width, effective_seam_width, padding))
        
        # Vertical seams - target exact tile boundaries
        for x in range(tile_width, width, tile_width):
            for y in range(0, height, tile_height):
                # Center seam exactly on tile boundary
                seam_x = x - effective_seam_width // 2
                tiles.append(self._get_tile_info(image, seam_x, y, effective_seam_width, tile_height, padding))
        
        return tiles

    def _generate_targeted_intersection_tiles(self, image, tile_width, tile_height, seam_width, padding):
        """Generate intersection tiles with precise targeting"""
        width, height = image.size
        tiles = []
        
        # Smaller intersection areas for precision
        effective_seam_width = max(seam_width // 2, 32)
        
        # Intersection points where seams cross
        for y in range(tile_height, height, tile_height):
            for x in range(tile_width, width, tile_width):
                # Center intersection exactly on tile boundaries
                intersection_x = x - effective_seam_width // 2
                intersection_y = y - effective_seam_width // 2
                tiles.append(self._get_tile_info(image, intersection_x, intersection_y, effective_seam_width, effective_seam_width, padding))
        
        return tiles

    def _generate_seam_tiles(self, image, tile_width, tile_height, seam_width, padding):
        width, height = image.size
        tiles = []
        
        # Horizontal seams
        for y in range(tile_height, height, tile_height):
            for x in range(0, width, tile_width):
                tiles.append(self._get_tile_info(image, x, y - seam_width // 2, tile_width, seam_width, padding))
        
        # Vertical seams        
        for x in range(tile_width, width, tile_width):
            for y in range(0, height, tile_height):
                tiles.append(self._get_tile_info(image, x - seam_width // 2, y, seam_width, tile_height, padding))
        return tiles

    def _generate_intersection_tiles(self, image, tile_width, tile_height, seam_width, padding):
        width, height = image.size
        tiles = []
        
        # Intersection points where horizontal and vertical seams meet
        for y in range(tile_height, height, tile_height):
            for x in range(tile_width, width, tile_width):
                tiles.append(self._get_tile_info(image, x - seam_width // 2, y - seam_width // 2, seam_width, seam_width, padding))
        return tiles

NODE_CLASS_MAPPINGS = {
    "UltimateSeedVR2Upscaler": UltimateSeedVR2Upscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSeedVR2Upscaler": "Ultimate SeedVR2 Upscaler"
}
