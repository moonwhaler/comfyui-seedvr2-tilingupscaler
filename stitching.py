"""Stitching algorithms for blending upscaled tiles."""

import torch
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import convolve2d
from collections import defaultdict
from typing import Any, Dict, List, Optional

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not available. Bilateral filtering will use Gaussian approximation.")

from .image_utils import tensor_to_pil, pil_to_tensor
from .seedvr2_adapter import execute_seedvr2


def _snap_to_4n_plus_1(value: int) -> int:
    """Snap a batch size down to the largest 4n+1 value (1, 5, 9, 13, 17, 21...) not exceeding it.

    SeedVR2's temporal VAE requires frame counts following the 4n+1 pattern.
    """
    if value <= 1:
        return 1
    n = (value - 1) // 4
    return 4 * n + 1


def _resolve_batch_size(requested: int, remaining: int) -> int:
    """Resolve the batch size for the next sub-batch.

    Never exceeds the user-requested batch size, and never exceeds the number of
    tiles left to process. The result is snapped down to the 4n+1 pattern required
    by SeedVR2.
    """
    return _snap_to_4n_plus_1(min(max(requested, 1), remaining))


def _resolution_for_tile(tile_width: int, tile_height: int, upscale_factor: float, cap: int) -> int:
    """Resolution for the tile's shortest edge that reproduces the global
    upscale factor, capped by the user's VRAM/quality ceiling.

    Without this, every tile is sent to SeedVR2 with the same literal
    `cap` value regardless of its own aspect ratio, so leftover-remainder
    edge tiles (much narrower than the main grid) get scaled far more
    aggressively than the rest of the image.
    """
    shortest_edge = min(tile_width, tile_height)
    target = min(shortest_edge * upscale_factor, cap)
    return max(min(64, cap), int(round(target)))


def _create_base_image(
    original_image: Image.Image,
    width: int,
    height: int,
    dit_config: Dict[str, Any],
    vae_config: Dict[str, Any],
    seed: int,
    tile_upscale_resolution: int,
    color_correction: str = "lab",
) -> Image.Image:
    """Create base image for stitching by upscaling the original at low resolution."""
    base_tensor = pil_to_tensor(original_image)
    base_upscaled = execute_seedvr2(
        images=base_tensor,
        dit_config=dit_config,
        vae_config=vae_config,
        seed=seed,
        resolution=min(512, tile_upscale_resolution // 2),
        batch_size=1,
        color_correction=color_correction,
    )
    base_pil = tensor_to_pil(base_upscaled)
    return base_pil.resize((width, height), Image.LANCZOS)


def _batch_upscale_tiles(
    tiles: List[Dict],
    dit_config: Dict[str, Any],
    vae_config: Dict[str, Any],
    seed: int,
    tile_upscale_resolution: int,
    upscale_factor: float,
    progress=None,
    color_correction: str = "lab",
    tile_batch_size: int = 1,
) -> List[Image.Image]:
    """Batch process tiles by grouping them by size for optimal performance.

    Args:
        upscale_factor: Global output/input scale factor. Each size group's
            SeedVR2 resolution is derived from its own shortest edge times
            this factor (capped by tile_upscale_resolution) so every tile is
            upscaled by a consistent amount regardless of its aspect ratio.
        tile_batch_size: Maximum number of tiles sent to SeedVR2 in a single call.
            1 processes tiles one at a time (lowest VRAM). Values above 1 are
            snapped down to the 4n+1 pattern SeedVR2 requires.
    """
    # Group tiles by their dimensions
    tiles_by_size = defaultdict(list)
    for idx, tile_info in enumerate(tiles):
        tile_size = (tile_info["tile"].width, tile_info["tile"].height)
        tiles_by_size[tile_size].append((idx, tile_info))

    # Process each size group with optimal batch sizes
    upscaled_tiles = [None] * len(tiles)
    tiles_processed_count = 0

    for tile_size, tile_group in tiles_by_size.items():
        num_tiles_in_group = len(tile_group)
        processed_tiles = 0

        # Process this size group in optimal sub-batches
        while processed_tiles < num_tiles_in_group:
            remaining = num_tiles_in_group - processed_tiles
            batch_size = _resolve_batch_size(tile_batch_size, remaining)

            # Get tiles for this sub-batch
            sub_batch = tile_group[processed_tiles:processed_tiles + batch_size]

            # Collect tensors for this sub-batch
            tile_tensors = [pil_to_tensor(tile_info["tile"]) for _, tile_info in sub_batch]
            batch_tensor = torch.cat(tile_tensors, dim=0)

            # Process this sub-batch
            group_resolution = _resolution_for_tile(tile_size[0], tile_size[1], upscale_factor, tile_upscale_resolution)
            upscaled_batch = execute_seedvr2(
                images=batch_tensor,
                dit_config=dit_config,
                vae_config=vae_config,
                seed=seed,
                resolution=group_resolution,
                batch_size=batch_size,
                color_correction=color_correction,
            )

            # Store results back in original order
            for batch_idx, (original_idx, _) in enumerate(sub_batch):
                upscaled_tiles[original_idx] = tensor_to_pil(upscaled_batch[batch_idx:batch_idx+1])
                tiles_processed_count += 1
                # Update progress after each tile
                if progress:
                    progress.update_sub_progress(f"AI Upscaling ({tiles_processed_count}/{len(tiles)})", 1)

            processed_tiles += batch_size

    return upscaled_tiles


def _prepare_tile_for_stitching(tile_info: Dict, ai_upscaled_tile: Image.Image, upscale_factor: float) -> Dict:
    """Prepare an upscaled tile for stitching by resizing, positioning, and cropping.

    This function handles both regular overlap padding (for blending adjacent tiles)
    and memory padding (added for GPU efficiency). Memory padding must be completely
    removed as it contains reflected/extended content that shouldn't be in the output.
    """
    # Get the original tile size (before memory padding was added)
    original_width, original_height = tile_info["original_tile_size"]

    # The AI upscaled the full tile (including memory padding)
    # We need to first crop out the memory padding, then handle overlap padding

    # Get memory padding info
    mem_left_pad, mem_top_pad, mem_right_pad, mem_bottom_pad = tile_info.get("memory_padding", (0, 0, 0, 0))

    # Calculate the upscaled dimensions
    # The upscaled tile corresponds to the full padded tile (with memory padding)
    full_tile_width = tile_info["tile"].width
    full_tile_height = tile_info["tile"].height

    # Resize the upscaled tile to match the expected output size for the full tile
    target_full_width = int(full_tile_width * upscale_factor)
    target_full_height = int(full_tile_height * upscale_factor)
    resized_tile = ai_upscaled_tile.resize((target_full_width, target_full_height), Image.LANCZOS)

    # First, crop out the memory padding (scale the memory padding amounts)
    scaled_mem_right = int(mem_right_pad * upscale_factor)
    scaled_mem_bottom = int(mem_bottom_pad * upscale_factor)

    # The original content (without memory padding) is in the top-left portion
    original_content_width = int(original_width * upscale_factor)
    original_content_height = int(original_height * upscale_factor)

    # Crop to remove memory padding - keep only the original content area
    if scaled_mem_right > 0 or scaled_mem_bottom > 0:
        resized_tile = resized_tile.crop((0, 0, original_content_width, original_content_height))

    # Now handle the regular overlap padding for blending
    # Calculate positioning
    paste_x = int(tile_info["position"][0] * upscale_factor)
    paste_y = int(tile_info["position"][1] * upscale_factor)
    final_tile_width = int(tile_info["actual_size"][0] * upscale_factor)
    final_tile_height = int(tile_info["actual_size"][1] * upscale_factor)

    # Calculate scaled overlap padding
    left_pad, top_pad, right_pad, bottom_pad = tile_info["padding"]

    scaled_left_pad = int(left_pad * upscale_factor)
    scaled_top_pad = int(top_pad * upscale_factor)
    scaled_right_pad = int(right_pad * upscale_factor)
    scaled_bottom_pad = int(bottom_pad * upscale_factor)

    # Keep half the padding on ALL sides to create overlap for blending
    keep_left = scaled_left_pad // 2 if left_pad > 0 else 0
    keep_top = scaled_top_pad // 2 if top_pad > 0 else 0
    keep_right = scaled_right_pad // 2 if right_pad > 0 else 0
    keep_bottom = scaled_bottom_pad // 2 if bottom_pad > 0 else 0

    # Crop the tile - keep partial overlap padding on all sides for blending
    crop_box = (
        scaled_left_pad - keep_left,
        scaled_top_pad - keep_top,
        min(scaled_left_pad + final_tile_width + keep_right, resized_tile.width),
        min(scaled_top_pad + final_tile_height + keep_bottom, resized_tile.height)
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


def _build_laplacian_pyramid(image: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """Build a Laplacian pyramid for multi-band blending.

    Args:
        image: numpy array (H, W, C)
        levels: number of pyramid levels

    Returns:
        List of Laplacian pyramid levels (finest to coarsest)
    """
    gaussian_pyramid = [image.astype(np.float64)]

    # Build Gaussian pyramid
    for i in range(levels):
        down = ndimage.zoom(gaussian_pyramid[-1], (0.5, 0.5, 1), order=1)
        gaussian_pyramid.append(down)

    # Build Laplacian pyramid
    laplacian_pyramid = []
    for i in range(levels):
        # Upscale the next level
        size = gaussian_pyramid[i].shape
        upscaled = ndimage.zoom(gaussian_pyramid[i + 1],
                               (size[0] / gaussian_pyramid[i + 1].shape[0],
                                size[1] / gaussian_pyramid[i + 1].shape[1],
                                1), order=1)
        # Laplacian = Gaussian - upscaled(next_gaussian)
        laplacian = gaussian_pyramid[i] - upscaled
        laplacian_pyramid.append(laplacian)

    # Add the smallest Gaussian as the last level
    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid


def _collapse_laplacian_pyramid(laplacian_pyramid: List[np.ndarray]) -> np.ndarray:
    """Collapse a Laplacian pyramid back to an image.

    Args:
        laplacian_pyramid: List of Laplacian levels (finest to coarsest)

    Returns:
        Reconstructed image as numpy array
    """
    # Start with the coarsest level
    image = laplacian_pyramid[-1]

    # Reconstruct from coarse to fine
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        # Upscale current image to match next level size
        size = laplacian_pyramid[i].shape
        upscaled = ndimage.zoom(image,
                               (size[0] / image.shape[0],
                                size[1] / image.shape[1],
                                1), order=1)
        # Add the Laplacian details
        image = upscaled + laplacian_pyramid[i]

    return image


def _apply_bilateral_filter(image, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filtering for edge-preserving smoothing.

    Args:
        image: PIL Image or numpy array
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Filtered image as numpy array
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    if HAS_OPENCV:
        # Use OpenCV's optimized bilateral filter
        filtered = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)
    else:
        # Fallback: Use Gaussian approximation
        filtered = img_array.astype(np.float64)

        # Apply edge-aware smoothing using gradients
        for channel in range(3):
            channel_data = filtered[:, :, channel]

            # Detect edges
            sobel_x = ndimage.sobel(channel_data, axis=1)
            sobel_y = ndimage.sobel(channel_data, axis=0)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

            # Normalize and invert (smooth where no edges)
            if edge_magnitude.max() > 0:
                edge_magnitude = edge_magnitude / edge_magnitude.max()
            smoothing_weight = 1.0 - edge_magnitude

            # Apply adaptive Gaussian smoothing
            smoothed = ndimage.gaussian_filter(channel_data, sigma=sigma_space / 10.0)
            filtered[:, :, channel] = channel_data * edge_magnitude + smoothed * smoothing_weight

        filtered = np.clip(filtered, 0, 255).astype(np.uint8)

    return filtered


def _compute_structure_tensor(image: np.ndarray, sigma: float = 1.5):
    """Compute structure tensor for content-aware blending.

    Args:
        image: numpy array (H, W, C)
        sigma: Gaussian smoothing sigma for structure tensor

    Returns:
        edge_strength: Edge strength map (H, W)
        coherence: Local structure coherence (H, W)
    """
    # Convert to grayscale for structure analysis
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # Compute gradients
    Ix = ndimage.sobel(gray, axis=1)
    Iy = ndimage.sobel(gray, axis=0)

    # Compute structure tensor components
    Ixx = ndimage.gaussian_filter(Ix * Ix, sigma)
    Iyy = ndimage.gaussian_filter(Iy * Iy, sigma)
    Ixy = ndimage.gaussian_filter(Ix * Iy, sigma)

    # Compute eigenvalues for edge strength and coherence
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy

    # Eigenvalues: lambda = (trace ± sqrt(trace^2 - 4*det)) / 2
    discriminant = np.maximum(trace * trace - 4 * det, 0)
    lambda1 = (trace + np.sqrt(discriminant)) / 2
    lambda2 = (trace - np.sqrt(discriminant)) / 2

    # Edge strength (larger eigenvalue)
    edge_strength = lambda1

    # Coherence (anisotropy measure)
    coherence = np.zeros_like(trace)
    mask = lambda1 > 1e-5
    coherence[mask] = (lambda1[mask] - lambda2[mask]) / (lambda1[mask] + lambda2[mask])

    # Normalize
    if edge_strength.max() > 0:
        edge_strength = edge_strength / edge_strength.max()
    coherence = np.clip(coherence, 0, 1)

    return edge_strength, coherence


def process_and_stitch(
    tiles: List[Dict],
    width: int,
    height: int,
    dit_config: Dict[str, Any],
    vae_config: Dict[str, Any],
    seed: int,
    tile_upscale_resolution: int,
    upscale_factor: float,
    mask_blur: int,
    progress,
    original_image: Image.Image,
    anti_aliasing_strength: float = 0.0,
    blending_method: str = "auto",
    color_correction: str = "lab",
    tile_batch_size: int = 1,
) -> Image.Image:
    """Main stitching function that chooses the appropriate method based on settings.

    Args:
        tiles: List of tile info dictionaries
        width: Output image width
        height: Output image height
        dit_config: DiT model configuration
        vae_config: VAE model configuration
        seed: Random seed
        tile_upscale_resolution: Resolution for upscaling tiles
        upscale_factor: Scale factor for output
        mask_blur: Blur radius for mask blending
        progress: Progress tracker
        original_image: Original input image
        anti_aliasing_strength: Anti-aliasing strength (0-1)
        blending_method: Blending method to use
        color_correction: Color correction method for SeedVR2
        tile_batch_size: Maximum tiles per SeedVR2 call (1 = lowest VRAM usage)

    Returns:
        Stitched output image
    """
    # Auto mode: choose based on mask_blur value
    if blending_method == "auto":
        if mask_blur == 0:
            blending_method = "simple"
        elif mask_blur <= 2:
            blending_method = "linear"
        else:
            blending_method = "linear"

    print(f"Using {blending_method} blending method...")

    # Common kwargs for all blending methods
    kwargs = {
        "tiles": tiles,
        "width": width,
        "height": height,
        "dit_config": dit_config,
        "vae_config": vae_config,
        "seed": seed,
        "tile_upscale_resolution": tile_upscale_resolution,
        "upscale_factor": upscale_factor,
        "progress": progress,
        "original_image": original_image,
        "color_correction": color_correction,
        "tile_batch_size": tile_batch_size,
    }

    # Route to appropriate blending function
    if blending_method == "multiband":
        result = _process_and_stitch_multiband(**kwargs)
    elif blending_method == "bilateral":
        result = _process_and_stitch_bilateral(**kwargs, mask_blur=mask_blur)
    elif blending_method == "content_aware":
        result = _process_and_stitch_content_aware(**kwargs, mask_blur=mask_blur)
    elif blending_method == "simple":
        result = _process_and_stitch_zero_blur(**kwargs)
    else:  # "linear" or default
        result = _process_and_stitch_blended(**kwargs, mask_blur=mask_blur)

    # Apply anti-aliasing if requested
    if anti_aliasing_strength > 0:
        result = _apply_edge_aware_antialiasing(result, anti_aliasing_strength)

    return result


def _apply_edge_aware_antialiasing(image: Image.Image, strength: float) -> Image.Image:
    """Apply edge-aware anti-aliasing using Sobel edge detection."""
    img_array = np.array(image, dtype=np.float64)
    smoothed = np.zeros_like(img_array)

    for channel in range(3):
        channel_data = img_array[:, :, channel]

        # Apply Sobel filters to detect edges
        sobel_x = ndimage.sobel(channel_data, axis=1)
        sobel_y = ndimage.sobel(channel_data, axis=0)

        # Calculate edge magnitude
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize edge magnitude to 0-1
        if edge_magnitude.max() > 0:
            edge_magnitude = edge_magnitude / edge_magnitude.max()

        # Create inverse edge map
        smoothing_mask = 1.0 - edge_magnitude
        smoothing_mask = 1.0 - (smoothing_mask * strength)

        # Apply Gaussian smoothing
        sigma = 0.5 + (strength * 1.5)
        smoothed_channel = ndimage.gaussian_filter(channel_data, sigma=sigma)

        # Selective blend
        smoothed[:, :, channel] = channel_data * smoothing_mask + smoothed_channel * (1.0 - smoothing_mask)

    smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
    return Image.fromarray(smoothed)


def _process_and_stitch_multiband(
    tiles: List[Dict],
    width: int,
    height: int,
    dit_config: Dict[str, Any],
    vae_config: Dict[str, Any],
    seed: int,
    tile_upscale_resolution: int,
    upscale_factor: float,
    progress,
    original_image: Image.Image,
    color_correction: str = "lab",
    tile_batch_size: int = 1,
) -> Image.Image:
    """Multi-band blending using Laplacian pyramids for frequency-separated stitching."""
    # Create base image
    base_image = _create_base_image(
        original_image, width, height, dit_config, vae_config, seed, tile_upscale_resolution, color_correction
    )

    # Batch process and upscale tiles
    upscaled_tiles = _batch_upscale_tiles(
        tiles=tiles,
        dit_config=dit_config,
        vae_config=vae_config,
        seed=seed,
        tile_upscale_resolution=tile_upscale_resolution,
        upscale_factor=upscale_factor,
        progress=progress,
        color_correction=color_correction,
        tile_batch_size=tile_batch_size,
    )

    # Build Laplacian pyramid for base image
    base_array = np.array(base_image, dtype=np.float64)
    pyramid_levels = 4
    output_pyramid = _build_laplacian_pyramid(base_array, pyramid_levels)

    # Process each tile and blend into pyramid
    for tile_idx, tile_info in enumerate(tiles):
        ai_upscaled_tile = upscaled_tiles[tile_idx]

        progress.update_sub_progress("Resizing & Positioning", 2)
        prepared = _prepare_tile_for_stitching(tile_info, ai_upscaled_tile, upscale_factor)
        cropped_tile = prepared["cropped_tile"]
        paste_x_adjusted = prepared["paste_x"]
        paste_y_adjusted = prepared["paste_y"]

        progress.update_sub_progress("Multi-band Blending", 3)

        # Build Laplacian pyramid for this tile
        tile_array = np.array(cropped_tile, dtype=np.float64)

        # Create full-size tile array
        full_tile_array = np.zeros((height, width, 3), dtype=np.float64)
        end_x = min(paste_x_adjusted + tile_array.shape[1], width)
        end_y = min(paste_y_adjusted + tile_array.shape[0], height)

        tile_height = end_y - paste_y_adjusted
        tile_width = end_x - paste_x_adjusted
        full_tile_array[paste_y_adjusted:end_y, paste_x_adjusted:end_x] = tile_array[:tile_height, :tile_width]

        # Build pyramid for this tile
        tile_pyramid = _build_laplacian_pyramid(full_tile_array, pyramid_levels)

        # Create blending mask
        mask = np.zeros((height, width), dtype=np.float64)
        mask[paste_y_adjusted:end_y, paste_x_adjusted:end_x] = 1.0

        # Apply feathering to mask based on overlap
        feather_size = min(32, tile_width // 4, tile_height // 4)
        if feather_size > 0:
            for y in range(paste_y_adjusted, end_y):
                for x in range(paste_x_adjusted, end_x):
                    dist_x = min(x - paste_x_adjusted, end_x - 1 - x)
                    dist_y = min(y - paste_y_adjusted, end_y - 1 - y)
                    dist = min(dist_x, dist_y)
                    if dist < feather_size:
                        mask[y, x] = dist / feather_size

        # Build pyramid for mask
        mask_pyramid = []
        current_mask = mask
        for i in range(pyramid_levels + 1):
            mask_pyramid.append(current_mask)
            if i < pyramid_levels:
                current_mask = ndimage.zoom(current_mask, 0.5, order=1)

        # Blend each level of the pyramid
        blended_pyramid = []
        for level in range(len(tile_pyramid)):
            mask_level = mask_pyramid[level]
            level_height, level_width = output_pyramid[level].shape[:2]
            if mask_level.shape[0] != level_height or mask_level.shape[1] != level_width:
                mask_level = ndimage.zoom(mask_level,
                                         (level_height / mask_level.shape[0],
                                          level_width / mask_level.shape[1]), order=1)

            mask_level = mask_level[:, :, np.newaxis]
            blended = output_pyramid[level] * (1 - mask_level) + tile_pyramid[level] * mask_level
            blended_pyramid.append(blended)

        output_pyramid = blended_pyramid
        progress.update()

    # Collapse pyramid to final image
    output_array = _collapse_laplacian_pyramid(output_pyramid)
    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    return Image.fromarray(output_array)


def _process_and_stitch_bilateral(
    tiles: List[Dict],
    width: int,
    height: int,
    dit_config: Dict[str, Any],
    vae_config: Dict[str, Any],
    seed: int,
    tile_upscale_resolution: int,
    upscale_factor: float,
    mask_blur: int,
    progress,
    original_image: Image.Image,
    color_correction: str = "lab",
    tile_batch_size: int = 1,
) -> Image.Image:
    """Bilateral filtering-based stitching for edge-preserving blending."""
    # Create base image
    base_image = _create_base_image(
        original_image, width, height, dit_config, vae_config, seed, tile_upscale_resolution, color_correction
    )

    output_image = base_image.copy()
    output_array = np.array(output_image, dtype=np.float64)
    weight_array = np.zeros((height, width), dtype=np.float64)

    # Batch process and upscale tiles
    upscaled_tiles = _batch_upscale_tiles(
        tiles=tiles,
        dit_config=dit_config,
        vae_config=vae_config,
        seed=seed,
        tile_upscale_resolution=tile_upscale_resolution,
        upscale_factor=upscale_factor,
        progress=progress,
        color_correction=color_correction,
        tile_batch_size=tile_batch_size,
    )

    # Process each tile
    for tile_idx, tile_info in enumerate(tiles):
        ai_upscaled_tile = upscaled_tiles[tile_idx]

        progress.update_sub_progress("Resizing & Positioning", 2)
        prepared = _prepare_tile_for_stitching(tile_info, ai_upscaled_tile, upscale_factor)
        cropped_tile = prepared["cropped_tile"]
        paste_x_adjusted = prepared["paste_x"]
        paste_y_adjusted = prepared["paste_y"]

        progress.update_sub_progress("Bilateral Filtering", 3)

        # Apply bilateral filter to tile
        tile_array = _apply_bilateral_filter(cropped_tile, d=9, sigma_color=75, sigma_space=75)
        tile_array = tile_array.astype(np.float64)

        # Define region
        end_x = min(paste_x_adjusted + tile_array.shape[1], width)
        end_y = min(paste_y_adjusted + tile_array.shape[0], height)

        # Blend with weighted averaging
        for y in range(paste_y_adjusted, end_y):
            for x in range(paste_x_adjusted, end_x):
                tile_x = x - paste_x_adjusted
                tile_y = y - paste_y_adjusted

                if tile_y < tile_array.shape[0] and tile_x < tile_array.shape[1]:
                    dist_x = min(tile_x, tile_array.shape[1] - 1 - tile_x)
                    dist_y = min(tile_y, tile_array.shape[0] - 1 - tile_y)
                    tile_weight = min(dist_x, dist_y) / max(tile_array.shape[1], tile_array.shape[0])
                    tile_weight = max(0.1, tile_weight)

                    current_weight = weight_array[y, x]
                    new_weight = current_weight + tile_weight

                    if current_weight > 0:
                        output_array[y, x] = (output_array[y, x] * current_weight + tile_array[tile_y, tile_x] * tile_weight) / new_weight
                    else:
                        output_array[y, x] = tile_array[tile_y, tile_x]

                    weight_array[y, x] = new_weight

        progress.update()

    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    return Image.fromarray(output_array)


def _process_and_stitch_content_aware(
    tiles: List[Dict],
    width: int,
    height: int,
    dit_config: Dict[str, Any],
    vae_config: Dict[str, Any],
    seed: int,
    tile_upscale_resolution: int,
    upscale_factor: float,
    mask_blur: int,
    progress,
    original_image: Image.Image,
    color_correction: str = "lab",
    tile_batch_size: int = 1,
) -> Image.Image:
    """Content-aware stitching using structure tensor for adaptive blending."""
    # Create base image
    base_image = _create_base_image(
        original_image, width, height, dit_config, vae_config, seed, tile_upscale_resolution, color_correction
    )

    output_array = np.array(base_image, dtype=np.float64)
    weight_array = np.zeros((height, width), dtype=np.float64)

    # Compute global structure for base image
    base_edge_strength, base_coherence = _compute_structure_tensor(output_array)

    # Batch process and upscale tiles
    upscaled_tiles = _batch_upscale_tiles(
        tiles=tiles,
        dit_config=dit_config,
        vae_config=vae_config,
        seed=seed,
        tile_upscale_resolution=tile_upscale_resolution,
        upscale_factor=upscale_factor,
        progress=progress,
        color_correction=color_correction,
        tile_batch_size=tile_batch_size,
    )

    # Process each tile
    for tile_idx, tile_info in enumerate(tiles):
        ai_upscaled_tile = upscaled_tiles[tile_idx]

        progress.update_sub_progress("Resizing & Positioning", 2)
        prepared = _prepare_tile_for_stitching(tile_info, ai_upscaled_tile, upscale_factor)
        cropped_tile = prepared["cropped_tile"]
        paste_x_adjusted = prepared["paste_x"]
        paste_y_adjusted = prepared["paste_y"]

        progress.update_sub_progress("Content-Aware Blending", 3)

        tile_array = np.array(cropped_tile, dtype=np.float64)
        tile_edge_strength, tile_coherence = _compute_structure_tensor(tile_array)

        end_x = min(paste_x_adjusted + tile_array.shape[1], width)
        end_y = min(paste_y_adjusted + tile_array.shape[0], height)

        for y in range(paste_y_adjusted, end_y):
            for x in range(paste_x_adjusted, end_x):
                tile_x = x - paste_x_adjusted
                tile_y = y - paste_y_adjusted

                if tile_y < tile_array.shape[0] and tile_x < tile_array.shape[1]:
                    local_edge = base_edge_strength[y, x]
                    local_coherence = base_coherence[y, x]
                    tile_edge = tile_edge_strength[tile_y, tile_x]

                    base_weight = 1.0 - local_coherence * 0.5
                    tile_weight = 1.0 - local_coherence * 0.5

                    if tile_edge > local_edge:
                        tile_weight *= (1.0 + (tile_edge - local_edge))

                    dist_x = min(tile_x, tile_array.shape[1] - 1 - tile_x)
                    dist_y = min(tile_y, tile_array.shape[0] - 1 - tile_y)
                    dist_weight = min(dist_x, dist_y) / max(tile_array.shape[1], tile_array.shape[0])
                    tile_weight *= max(0.1, dist_weight)

                    current_weight = weight_array[y, x]
                    new_weight = current_weight + tile_weight

                    if current_weight > 0:
                        output_array[y, x] = (output_array[y, x] * current_weight + tile_array[tile_y, tile_x] * tile_weight) / new_weight
                    else:
                        output_array[y, x] = tile_array[tile_y, tile_x]

                    weight_array[y, x] = new_weight

        progress.update()

    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    return Image.fromarray(output_array)


def _process_and_stitch_zero_blur(
    tiles: List[Dict],
    width: int,
    height: int,
    dit_config: Dict[str, Any],
    vae_config: Dict[str, Any],
    seed: int,
    tile_upscale_resolution: int,
    upscale_factor: float,
    progress,
    original_image: Image.Image,
    color_correction: str = "lab",
    tile_batch_size: int = 1,
) -> Image.Image:
    """Zero-blur stitching that preserves maximum detail through precise pixel averaging."""
    # Create base image
    base_image = _create_base_image(
        original_image, width, height, dit_config, vae_config, seed, tile_upscale_resolution, color_correction
    )

    output_array = np.array(base_image, dtype=np.float64)
    weight_array = np.zeros((height, width), dtype=np.float64)

    # Batch process and upscale tiles
    upscaled_tiles = _batch_upscale_tiles(
        tiles=tiles,
        dit_config=dit_config,
        vae_config=vae_config,
        seed=seed,
        tile_upscale_resolution=tile_upscale_resolution,
        upscale_factor=upscale_factor,
        progress=progress,
        color_correction=color_correction,
        tile_batch_size=tile_batch_size,
    )

    # Process each upscaled tile for stitching
    for tile_idx, tile_info in enumerate(tiles):
        ai_upscaled_tile = upscaled_tiles[tile_idx]

        progress.update_sub_progress("Resizing & Positioning", 2)
        prepared = _prepare_tile_for_stitching(tile_info, ai_upscaled_tile, upscale_factor)
        cropped_tile = prepared["cropped_tile"]
        paste_x_adjusted = prepared["paste_x"]
        paste_y_adjusted = prepared["paste_y"]

        tile_array = np.array(cropped_tile, dtype=np.float64)

        progress.update_sub_progress("Seamless Blending", 3)

        end_x = min(paste_x_adjusted + tile_array.shape[1], width)
        end_y = min(paste_y_adjusted + tile_array.shape[0], height)

        for y in range(paste_y_adjusted, end_y):
            for x in range(paste_x_adjusted, end_x):
                tile_x = x - paste_x_adjusted
                tile_y = y - paste_y_adjusted

                if tile_y < tile_array.shape[0] and tile_x < tile_array.shape[1]:
                    current_weight = weight_array[y, x]
                    new_weight = current_weight + 1.0

                    if current_weight > 0:
                        output_array[y, x] = (output_array[y, x] * current_weight + tile_array[tile_y, tile_x]) / new_weight
                    else:
                        output_array[y, x] = tile_array[tile_y, tile_x]

                    weight_array[y, x] = new_weight

        progress.update()

    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    return Image.fromarray(output_array)


def _process_and_stitch_blended(
    tiles: List[Dict],
    width: int,
    height: int,
    dit_config: Dict[str, Any],
    vae_config: Dict[str, Any],
    seed: int,
    tile_upscale_resolution: int,
    upscale_factor: float,
    mask_blur: int,
    progress,
    original_image: Image.Image,
    color_correction: str = "lab",
    tile_batch_size: int = 1,
) -> Image.Image:
    """Standard blended stitching with user-controlled blur."""
    # Create base image
    base_image = _create_base_image(
        original_image, width, height, dit_config, vae_config, seed, tile_upscale_resolution, color_correction
    )

    output_image = base_image.copy()

    # Batch process and upscale tiles
    upscaled_tiles = _batch_upscale_tiles(
        tiles=tiles,
        dit_config=dit_config,
        vae_config=vae_config,
        seed=seed,
        tile_upscale_resolution=tile_upscale_resolution,
        upscale_factor=upscale_factor,
        progress=progress,
        color_correction=color_correction,
        tile_batch_size=tile_batch_size,
    )

    # Process each upscaled tile for stitching
    for tile_idx, tile_info in enumerate(tiles):
        ai_upscaled_tile = upscaled_tiles[tile_idx]

        progress.update_sub_progress("Resizing & Positioning", 2)
        prepared = _prepare_tile_for_stitching(tile_info, ai_upscaled_tile, upscale_factor)
        cropped_tile = prepared["cropped_tile"]
        paste_x_adjusted = prepared["paste_x"]
        paste_y_adjusted = prepared["paste_y"]
        keep_left, keep_top, keep_right, keep_bottom = prepared["keep_padding"]

        progress.update_sub_progress("Mask Blending", 3)

        # Create mask with user-specified blur
        actual_crop_width = cropped_tile.width
        actual_crop_height = cropped_tile.height
        tile_mask = _create_precise_tile_mask(
            actual_crop_width, actual_crop_height, mask_blur,
            tile_info["padding"], keep_left, keep_top, keep_right, keep_bottom
        )

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

        progress.update()

    return output_image


def _create_precise_tile_mask(
    width: int,
    height: int,
    blur_radius: int,
    padding_info: tuple,
    keep_left: int = 0,
    keep_top: int = 0,
    keep_right: int = 0,
    keep_bottom: int = 0,
) -> Image.Image:
    """Create smart blending mask with proper overlap handling on all sides."""
    left_pad, top_pad, right_pad, bottom_pad = padding_info
    mask_array = np.full((height, width), 255, dtype=np.uint8)

    if blur_radius > 0:
        overlap_width_left = keep_left * 2 if keep_left > 0 else 0
        overlap_width_top = keep_top * 2 if keep_top > 0 else 0
        overlap_width_right = keep_right * 2 if keep_right > 0 else 0
        overlap_width_bottom = keep_bottom * 2 if keep_bottom > 0 else 0

        for y in range(height):
            for x in range(width):
                min_alpha = 255

                if left_pad > 0 and overlap_width_left > 0 and x < overlap_width_left:
                    fade_alpha = int(255 * x / overlap_width_left)
                    min_alpha = min(min_alpha, fade_alpha)

                if top_pad > 0 and overlap_width_top > 0 and y < overlap_width_top:
                    fade_alpha = int(255 * y / overlap_width_top)
                    min_alpha = min(min_alpha, fade_alpha)

                if right_pad > 0 and overlap_width_right > 0 and x >= width - overlap_width_right:
                    distance_from_overlap_start = x - (width - overlap_width_right)
                    fade_alpha = int(255 * (1.0 - distance_from_overlap_start / overlap_width_right))
                    min_alpha = min(min_alpha, fade_alpha)

                if bottom_pad > 0 and overlap_width_bottom > 0 and y >= height - overlap_width_bottom:
                    distance_from_overlap_start = y - (height - overlap_width_bottom)
                    fade_alpha = int(255 * (1.0 - distance_from_overlap_start / overlap_width_bottom))
                    min_alpha = min(min_alpha, fade_alpha)

                mask_array[y, x] = min_alpha

    return Image.fromarray(mask_array)
