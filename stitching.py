"""Stitching algorithms for blending upscaled tiles."""

import torch
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import convolve2d
from collections import defaultdict
from .image_utils import tensor_to_pil, pil_to_tensor
from .seedvr2_adapter import build_execute_kwargs


def _get_optimal_batch_size(num_tiles):
    """Calculate optimal batch size following 4n+1 pattern (1, 5, 9, 13, 17, 21...)"""
    if num_tiles <= 1:
        return 1
    # Find largest 4n+1 that doesn't exceed num_tiles
    n = (num_tiles - 1) // 4
    return 4 * n + 1


def _create_base_image(seedvr2_instance, width, height, model, seed, tile_upscale_resolution,
                       preserve_vram, block_swap_config, extra_args, base_image=None):
    """Create or return base image for stitching."""
    if base_image is not None:
        return base_image

    if hasattr(seedvr2_instance, '_original_image'):
        base_tensor = pil_to_tensor(seedvr2_instance._original_image)
        base_upscaled_tuple = _execute_seedvr2(
            seedvr2_instance,
            images=base_tensor,
            model=model,
            seed=seed,
            new_resolution=min(512, tile_upscale_resolution // 2),
            batch_size=1,
            preserve_vram=preserve_vram,
            block_swap_config=block_swap_config,
            extra_args=extra_args,
        )
        base_upscaled = tensor_to_pil(base_upscaled_tuple[0])
        return base_upscaled.resize((width, height), Image.LANCZOS)
    else:
        return Image.new('RGB', (width, height), (128, 128, 128))


def _batch_upscale_tiles(tiles, seedvr2_instance, model, seed, tile_upscale_resolution,
                         preserve_vram, block_swap_config, extra_args):
    """Batch process tiles by grouping them by size for optimal performance."""
    # Group tiles by their dimensions
    tiles_by_size = defaultdict(list)
    for idx, tile_info in enumerate(tiles):
        tile_size = (tile_info["tile"].width, tile_info["tile"].height)
        tiles_by_size[tile_size].append((idx, tile_info))

    # Process each size group with optimal batch sizes
    upscaled_tiles = [None] * len(tiles)  # Store results in original order

    for tile_size, tile_group in tiles_by_size.items():
        num_tiles_in_group = len(tile_group)
        processed_tiles = 0

        # Process this size group in optimal sub-batches
        while processed_tiles < num_tiles_in_group:
            remaining = num_tiles_in_group - processed_tiles
            batch_size = _get_optimal_batch_size(remaining)

            # Get tiles for this sub-batch
            sub_batch = tile_group[processed_tiles:processed_tiles + batch_size]

            # Collect tensors for this sub-batch
            tile_tensors = [pil_to_tensor(tile_info["tile"]) for _, tile_info in sub_batch]
            batch_tensor = torch.cat(tile_tensors, dim=0)

            # Process this sub-batch
            upscaled_batch_tuple = _execute_seedvr2(
                seedvr2_instance,
                images=batch_tensor,
                model=model,
                seed=seed,
                new_resolution=tile_upscale_resolution,
                batch_size=batch_size,
                preserve_vram=preserve_vram,
                block_swap_config=block_swap_config,
                extra_args=extra_args,
            )
            upscaled_batch = upscaled_batch_tuple[0]

            # Store results back in original order
            for batch_idx, (original_idx, _) in enumerate(sub_batch):
                upscaled_tiles[original_idx] = tensor_to_pil(upscaled_batch[batch_idx:batch_idx+1])

            processed_tiles += batch_size

    return upscaled_tiles


def _prepare_tile_for_stitching(tile_info, ai_upscaled_tile, upscale_factor):
    """Prepare an upscaled tile for stitching by resizing, positioning, and cropping."""
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
    scaled_right_pad = int(right_pad * upscale_factor)
    scaled_bottom_pad = int(bottom_pad * upscale_factor)
    scaled_mem_right_pad = int(mem_right_pad * upscale_factor)
    scaled_mem_bottom_pad = int(mem_bottom_pad * upscale_factor)

    # Keep half the padding on ALL sides to create overlap for blending
    keep_left = scaled_left_pad // 2 if left_pad > 0 else 0
    keep_top = scaled_top_pad // 2 if top_pad > 0 else 0
    keep_right = scaled_right_pad // 2 if right_pad > 0 else 0
    keep_bottom = scaled_bottom_pad // 2 if bottom_pad > 0 else 0

    # Crop the upscaled tile - keep partial padding on all sides
    crop_box = (
        scaled_left_pad - keep_left,
        scaled_top_pad - keep_top,
        scaled_left_pad + final_tile_width + keep_right,
        scaled_top_pad + final_tile_height + keep_bottom
    )
    cropped_tile = resized_tile.crop(crop_box)

    # Adjust paste position to account for kept left/top padding
    paste_x_adjusted = max(0, paste_x - keep_left)
    paste_y_adjusted = max(0, paste_y - keep_top)

    return {
        "cropped_tile": cropped_tile,
        "paste_x": paste_x_adjusted,
        "paste_y": paste_y_adjusted,
        "keep_padding": (keep_left, keep_top, keep_right, keep_bottom),
    }


def process_and_stitch(tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, mask_blur, progress, base_image=None, anti_aliasing_strength=0.0, extra_args=None):
    """Main stitching function that chooses the appropriate method based on blur setting."""
    if mask_blur == 0:
        print("Using zero-blur mode for maximum detail preservation...")
        result = process_and_stitch_zero_blur(tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, progress, base_image, extra_args=extra_args)
    else:
        result = process_and_stitch_blended(tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, mask_blur, progress, base_image, extra_args=extra_args)
    
    # Apply anti-aliasing if requested
    if anti_aliasing_strength > 0:
        result = apply_edge_aware_antialiasing(result, anti_aliasing_strength)
    
    return result


def apply_edge_aware_antialiasing(image, strength):
    """Apply edge-aware anti-aliasing using Sobel edge detection."""
    # Convert PIL image to numpy array
    img_array = np.array(image, dtype=np.float64)
    
    # Process each color channel
    smoothed = np.zeros_like(img_array)
    
    for channel in range(3):
        # Extract channel
        channel_data = img_array[:, :, channel]
        
        # Apply Sobel filters to detect edges
        sobel_x = ndimage.sobel(channel_data, axis=1)
        sobel_y = ndimage.sobel(channel_data, axis=0)
        
        # Calculate edge magnitude
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize edge magnitude to 0-1
        if edge_magnitude.max() > 0:
            edge_magnitude = edge_magnitude / edge_magnitude.max()
        
        # Create inverse edge map (1 where no edges, 0 where strong edges)
        # This ensures we smooth non-edge areas while preserving edges
        smoothing_mask = 1.0 - edge_magnitude
        
        # Apply adjustable strength
        smoothing_mask = 1.0 - (smoothing_mask * strength)
        
        # Apply Gaussian smoothing
        # Sigma scales with strength for adaptive smoothing
        sigma = 0.5 + (strength * 1.5)  # Range from 0.5 to 2.0
        smoothed_channel = ndimage.gaussian_filter(channel_data, sigma=sigma)
        
        # Selective blend: original in edge areas, smoothed in non-edge areas
        smoothed[:, :, channel] = channel_data * smoothing_mask + smoothed_channel * (1.0 - smoothing_mask)
    
    # Convert back to uint8 and PIL Image
    smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
    return Image.fromarray(smoothed)


def process_and_stitch_zero_blur(tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, progress, base_image=None, extra_args=None):
    """Zero-blur stitching that preserves maximum detail through precise pixel averaging."""
    # Create base image
    base_image = _create_base_image(
        seedvr2_instance, width, height, model, seed, tile_upscale_resolution,
        preserve_vram, block_swap_config, extra_args, base_image
    )

    # Use numpy arrays for precise pixel control
    output_array = np.array(base_image, dtype=np.float64)
    weight_array = np.zeros((height, width), dtype=np.float64)

    # Batch process and upscale tiles
    progress.update_sub_progress("AI Upscaling", 1)
    upscaled_tiles = _batch_upscale_tiles(
        tiles, seedvr2_instance, model, seed, tile_upscale_resolution,
        preserve_vram, block_swap_config, extra_args
    )

    # Now process each upscaled tile for stitching
    for tile_idx, tile_info in enumerate(tiles):
        # Get the pre-upscaled tile
        ai_upscaled_tile = upscaled_tiles[tile_idx]

        progress.update_sub_progress("Resizing & Positioning", 2)

        # Prepare tile for stitching
        prepared = _prepare_tile_for_stitching(tile_info, ai_upscaled_tile, upscale_factor)
        cropped_tile = prepared["cropped_tile"]
        paste_x_adjusted = prepared["paste_x"]
        paste_y_adjusted = prepared["paste_y"]

        tile_array = np.array(cropped_tile, dtype=np.float64)

        progress.update_sub_progress("Seamless Blending", 3)

        # Define the region in the output image
        end_x = min(paste_x_adjusted + tile_array.shape[1], width)
        end_y = min(paste_y_adjusted + tile_array.shape[0], height)

        # Pixel-perfect weighted averaging for seamless blending
        for y in range(paste_y_adjusted, end_y):
            for x in range(paste_x_adjusted, end_x):
                tile_x = x - paste_x_adjusted
                tile_y = y - paste_y_adjusted

                if 0 <= y < height and 0 <= x < width and tile_y < tile_array.shape[0] and tile_x < tile_array.shape[1]:
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


def process_and_stitch_blended(tiles, width, height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, mask_blur, progress, base_image=None, extra_args=None):
    """Standard blended stitching with user-controlled blur."""
    # Create base image
    base_image = _create_base_image(
        seedvr2_instance, width, height, model, seed, tile_upscale_resolution,
        preserve_vram, block_swap_config, extra_args, base_image
    )

    output_image = base_image.copy()

    # Batch process and upscale tiles
    progress.update_sub_progress("AI Upscaling", 1)
    upscaled_tiles = _batch_upscale_tiles(
        tiles, seedvr2_instance, model, seed, tile_upscale_resolution,
        preserve_vram, block_swap_config, extra_args
    )

    # Now process each upscaled tile for stitching
    for tile_idx, tile_info in enumerate(tiles):
        # Get the pre-upscaled tile
        ai_upscaled_tile = upscaled_tiles[tile_idx]

        progress.update_sub_progress("Resizing & Positioning", 2)

        # Prepare tile for stitching
        prepared = _prepare_tile_for_stitching(tile_info, ai_upscaled_tile, upscale_factor)
        cropped_tile = prepared["cropped_tile"]
        paste_x_adjusted = prepared["paste_x"]
        paste_y_adjusted = prepared["paste_y"]
        keep_left, keep_top, keep_right, keep_bottom = prepared["keep_padding"]

        progress.update_sub_progress("Mask Blending", 3)

        # Create mask with user-specified blur - use actual tile size including kept overlap on all sides
        actual_crop_width = cropped_tile.width
        actual_crop_height = cropped_tile.height
        tile_mask = create_precise_tile_mask(actual_crop_width, actual_crop_height, mask_blur, tile_info["padding"], keep_left, keep_top, keep_right, keep_bottom)

        # Create RGBA version of the tile for compositing
        tile_rgba = Image.new('RGBA', output_image.size, (0, 0, 0, 0))
        tile_rgba.paste(cropped_tile, (paste_x_adjusted, paste_y_adjusted))

        # Create full-size mask
        full_mask = Image.new('L', output_image.size, 0)
        full_mask.paste(tile_mask, (paste_x_adjusted, paste_y_adjusted))
        tile_rgba.putalpha(full_mask)
        
        # Alpha composite onto the output image
        output_rgba = output_image.convert('RGBA')
        output_rgba.alpha_composite(tile_rgba)
        output_image = output_rgba.convert('RGB')
        
        # Complete this tile
        progress.update()

    return output_image


def create_precise_tile_mask(width, height, blur_radius, padding_info, keep_left=0, keep_top=0, keep_right=0, keep_bottom=0):
    """Create smart blending mask with proper overlap handling on all sides.

    The mask ensures seamless blending in overlap regions by fading across the FULL overlap width.
    Since each tile keeps half the padding, the total overlap is keep_left + keep_right (which equals
    the full padding). Each tile must fade across this entire overlap region for masks to sum to 255.
    """
    left_pad, top_pad, right_pad, bottom_pad = padding_info
    mask_array = np.full((height, width), 255, dtype=np.uint8)

    if blur_radius > 0:
        # The overlap width is double the kept amount (keep_left on this tile + keep_right on neighbor)
        # We fade across the full overlap width to ensure masks sum to 255
        overlap_width_left = keep_left * 2 if keep_left > 0 else 0
        overlap_width_top = keep_top * 2 if keep_top > 0 else 0
        overlap_width_right = keep_right * 2 if keep_right > 0 else 0
        overlap_width_bottom = keep_bottom * 2 if keep_bottom > 0 else 0

        for y in range(height):
            for x in range(width):
                min_alpha = 255

                # LEFT EDGE: Fade from 0 to 255 across the FULL overlap width
                # Overlap region is [0, overlap_width_left), fade across entire width
                if left_pad > 0 and overlap_width_left > 0 and x < overlap_width_left:
                    # Fade from 0 to 255 across the full overlap width
                    # At x=0: fade_alpha = 0
                    # At x=keep_left-1: fade_alpha ≈ 128 (halfway)
                    # Note: we only have pixels up to keep_left in this tile
                    fade_alpha = int(255 * x / overlap_width_left)
                    min_alpha = min(min_alpha, fade_alpha)

                # TOP EDGE: Same logic as left edge
                if top_pad > 0 and overlap_width_top > 0 and y < overlap_width_top:
                    fade_alpha = int(255 * y / overlap_width_top)
                    min_alpha = min(min_alpha, fade_alpha)

                # RIGHT EDGE: Fade from 255 to 0 across the FULL overlap width
                # Overlap starts at (width - overlap_width_right), extends to width
                if right_pad > 0 and overlap_width_right > 0 and x >= width - overlap_width_right:
                    # Distance from start of overlap region
                    distance_from_overlap_start = x - (width - overlap_width_right)
                    # Fade from 255 to 0 across the full overlap width
                    fade_alpha = int(255 * (1.0 - distance_from_overlap_start / overlap_width_right))
                    min_alpha = min(min_alpha, fade_alpha)

                # BOTTOM EDGE: Same logic as right edge
                if bottom_pad > 0 and overlap_width_bottom > 0 and y >= height - overlap_width_bottom:
                    distance_from_overlap_start = y - (height - overlap_width_bottom)
                    fade_alpha = int(255 * (1.0 - distance_from_overlap_start / overlap_width_bottom))
                    min_alpha = min(min_alpha, fade_alpha)

                mask_array[y, x] = min_alpha
    
    return Image.fromarray(mask_array)


def _execute_seedvr2(seedvr2_instance, *, images, model, seed, new_resolution, batch_size, preserve_vram,
                     block_swap_config, extra_args=None):
    kwargs = build_execute_kwargs(
        seedvr2_instance,
        images=images,
        model=model,
        seed=seed,
        new_resolution=new_resolution,
        batch_size=batch_size,
        preserve_vram=preserve_vram,
        block_swap_config=block_swap_config,
        extra_args=extra_args,
    )
    return seedvr2_instance.execute(**kwargs)
