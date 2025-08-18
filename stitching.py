"""Stitching algorithms for blending upscaled tiles."""

import numpy as np
from PIL import Image
from .image_utils import tensor_to_pil, pil_to_tensor


def process_and_stitch(tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, mask_blur, progress, base_image=None):
    """Main stitching function that chooses the appropriate method based on blur setting."""
    if mask_blur == 0:
        print("Using zero-blur mode for maximum detail preservation...")
        return process_and_stitch_zero_blur(tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, progress, base_image)
    else:
        return process_and_stitch_blended(tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, mask_blur, progress, base_image)


def process_and_stitch_zero_blur(tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, progress, base_image=None):
    """Zero-blur stitching that preserves maximum detail through precise pixel averaging."""
    # Create base image if not provided
    if base_image is None:
        if hasattr(seedvr2_instance, '_original_image'):
            base_tensor = pil_to_tensor(seedvr2_instance._original_image)
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
    for tile_idx, tile_info in enumerate(tiles):
        progress.update_sub_progress("AI Upscaling", 1)
        
        # Upscale tile
        tile_tensor = pil_to_tensor(tile_info["tile"])
        upscaled_tile_tuple = seedvr2_instance.execute(images=tile_tensor, model=model, seed=seed, new_resolution=tile_upscale_resolution, batch_size=1, preserve_vram=preserve_vram, block_swap_config=block_swap_config)
        ai_upscaled_tile = tensor_to_pil(upscaled_tile_tuple[0])

        progress.update_sub_progress("Resizing & Positioning", 2)
        
        # Resize to final target size
        target_tile_width = int(tile_info["tile"].width * upscale_factor)
        target_tile_height = int(tile_info["tile"].height * upscale_factor)
        resized_tile = ai_upscaled_tile.resize((target_tile_width, target_tile_height), Image.LANCZOS)
        
        # Calculate positioning
        paste_x = int(tile_info["position"][0] * upscale_factor)
        paste_y = int(tile_info["position"][1] * upscale_factor)
        final_tile_width = int(tile_info["actual_size"][0] * upscale_factor)
        final_tile_height = int(tile_info["actual_size"][1] * upscale_factor)
        
        # Calculate scaled padding (both regular and memory padding)
        left_pad, top_pad, right_pad, bottom_pad = tile_info["padding"]
        mem_left_pad, mem_top_pad, mem_right_pad, mem_bottom_pad = tile_info.get("memory_padding", (0, 0, 0, 0))
        
        scaled_left_pad = int(left_pad * upscale_factor)
        scaled_top_pad = int(top_pad * upscale_factor)
        scaled_mem_right_pad = int(mem_right_pad * upscale_factor)
        scaled_mem_bottom_pad = int(mem_bottom_pad * upscale_factor)
        
        # Crop the upscaled tile to remove both regular and memory padding
        crop_box = (
            scaled_left_pad,
            scaled_top_pad,
            scaled_left_pad + final_tile_width,
            scaled_top_pad + final_tile_height
        )
        cropped_tile = resized_tile.crop(crop_box)
        tile_array = np.array(cropped_tile, dtype=np.float64)
        
        progress.update_sub_progress("Seamless Blending", 3)
        
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
        
        # Complete this tile
        progress.update()
    
    # Convert back to PIL Image
    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    return Image.fromarray(output_array)


def process_and_stitch_blended(tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, mask_blur, progress, base_image=None):
    """Standard blended stitching with user-controlled blur."""
    # Create base image if not provided
    if base_image is None:
        if hasattr(seedvr2_instance, '_original_image'):
            base_tensor = pil_to_tensor(seedvr2_instance._original_image)
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
    for tile_idx, tile_info in enumerate(tiles):
        progress.update_sub_progress("AI Upscaling", 1)
        
        # Upscale tile
        tile_tensor = pil_to_tensor(tile_info["tile"])
        upscaled_tile_tuple = seedvr2_instance.execute(images=tile_tensor, model=model, seed=seed, new_resolution=tile_upscale_resolution, batch_size=1, preserve_vram=preserve_vram, block_swap_config=block_swap_config)
        ai_upscaled_tile = tensor_to_pil(upscaled_tile_tuple[0])

        progress.update_sub_progress("Resizing & Positioning", 2)
        
        # Resize to final target size
        target_tile_width = int(tile_info["tile"].width * upscale_factor)
        target_tile_height = int(tile_info["tile"].height * upscale_factor)
        resized_tile = ai_upscaled_tile.resize((target_tile_width, target_tile_height), Image.LANCZOS)
        
        # Calculate the region this tile covers (without padding)
        paste_x = int(tile_info["position"][0] * upscale_factor)
        paste_y = int(tile_info["position"][1] * upscale_factor)
        final_tile_width = int(tile_info["actual_size"][0] * upscale_factor)
        final_tile_height = int(tile_info["actual_size"][1] * upscale_factor)
        
        # Calculate scaled padding to crop correctly (both regular and memory padding)
        left_pad, top_pad, right_pad, bottom_pad = tile_info["padding"]
        mem_left_pad, mem_top_pad, mem_right_pad, mem_bottom_pad = tile_info.get("memory_padding", (0, 0, 0, 0))
        
        scaled_left_pad = int(left_pad * upscale_factor)
        scaled_top_pad = int(top_pad * upscale_factor)
        scaled_mem_right_pad = int(mem_right_pad * upscale_factor)
        scaled_mem_bottom_pad = int(mem_bottom_pad * upscale_factor)
        
        # Crop the upscaled tile to remove both regular and memory padding
        crop_box = (
            scaled_left_pad,
            scaled_top_pad,
            scaled_left_pad + final_tile_width,
            scaled_top_pad + final_tile_height
        )
        cropped_tile = resized_tile.crop(crop_box)
        
        progress.update_sub_progress("Mask Blending", 3)
        
        # Create mask with user-specified blur - respects exact setting
        tile_mask = create_precise_tile_mask(final_tile_width, final_tile_height, mask_blur, tile_info["padding"])
        
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
        
        # Complete this tile
        progress.update()

    return output_image


def create_precise_tile_mask(width, height, blur_radius, padding_info):
    """Create smart blending mask - zero blur interior, minimal blur at seams."""
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
