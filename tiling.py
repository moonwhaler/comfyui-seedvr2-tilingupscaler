"""Tiling utilities for dividing images into overlapping tiles."""


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
