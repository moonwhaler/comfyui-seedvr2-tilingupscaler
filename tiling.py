"""Tiling utilities for dividing images into overlapping tiles."""


def calculate_efficient_tile_size(width, height):
    """Calculate GPU-efficient tile dimensions by padding to optimal sizes."""
    # Target minimum efficient size and prefer multiples of 16 for GPU optimization
    min_efficient_size = 512
    
    # Round up to next multiple of 16 that's at least min_efficient_size
    efficient_width = max(min_efficient_size, ((width + 15) // 16) * 16)
    efficient_height = max(min_efficient_size, ((height + 15) // 16) * 16)
    
    return efficient_width, efficient_height


def generate_tiles(image, tile_width, tile_height, padding, strategy):
    """Generate tiles based on the specified strategy."""
    width, height = image.size
    tiles = []
    
    if strategy == "Linear":
        for y in range(0, height, tile_height):
            for x in range(0, width, tile_width):
                tiles.append(get_tile_info(image, x, y, tile_width, tile_height, padding))
    elif strategy == "Chess":
        for y_idx, y in enumerate(range(0, height, tile_height)):
            for x_idx, x in enumerate(range(0, width, tile_width)):
                if (x_idx + y_idx) % 2 == 0:
                    tiles.append(get_tile_info(image, x, y, tile_width, tile_height, padding))
        for y_idx, y in enumerate(range(0, height, tile_height)):
            for x_idx, x in enumerate(range(0, width, tile_width)):
                if (x_idx + y_idx) % 2 != 0:
                    tiles.append(get_tile_info(image, x, y, tile_width, tile_height, padding))
    return tiles


def get_tile_info(image, x, y, tile_width, tile_height, padding):
    """Extract tile information and crop the tile with padding."""
    from PIL import Image
    
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
    
    # Calculate efficient dimensions for GPU processing
    current_width, current_height = tile.size
    efficient_width, efficient_height = calculate_efficient_tile_size(current_width, current_height)
    
    # Add memory padding if needed for GPU efficiency
    memory_pad_right = efficient_width - current_width
    memory_pad_bottom = efficient_height - current_height
    
    if memory_pad_right > 0 or memory_pad_bottom > 0:
        # Create GPU-efficient padded tile
        padded_tile = Image.new('RGB', (efficient_width, efficient_height), (128, 128, 128))
        padded_tile.paste(tile, (0, 0))
        tile = padded_tile
    
    return {
        "tile": tile,
        "position": (x, y),
        "actual_size": (actual_tile_width, actual_tile_height),
        "padding": (left_pad, top_pad, right_pad, bottom_pad),
        "memory_padding": (0, 0, memory_pad_right, memory_pad_bottom),
        "original_tile_size": (current_width, current_height),
    }
