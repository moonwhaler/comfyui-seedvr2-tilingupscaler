import torch
import numpy as np
from PIL import Image
import nodes

# Import new ComfyUI progress system
try:
    from server import PromptServer
    from comfy_execution.utils import get_executing_context
    from comfy_execution.progress import get_progress_state
    WEBSOCKET_PROGRESS_AVAILABLE = True
except ImportError:
    WEBSOCKET_PROGRESS_AVAILABLE = False
    print("WebSocket progress system not available - using console progress only")

# Fallback for older ComfyUI versions
try:
    import comfy.utils
    LEGACY_PROGRESS_AVAILABLE = True
except ImportError:
    LEGACY_PROGRESS_AVAILABLE = False

class Progress:
    def __init__(self, total_steps, sub_steps_per_tile=3):
        self.total_steps = total_steps
        self.sub_steps_per_tile = sub_steps_per_tile
        self.total_sub_steps = total_steps * sub_steps_per_tile
        self.current_step = 0
        self.current_sub_step = 0
        
        # Get execution context for WebSocket progress
        self.context = None
        self.node_id = None
        self.prompt_id = None
        
        if WEBSOCKET_PROGRESS_AVAILABLE:
            try:
                self.context = get_executing_context()
                if self.context:
                    self.node_id = self.context.node_id
                    self.prompt_id = self.context.prompt_id
            except Exception as e:
                print(f"Could not get execution context: {e}")
        
        print(f"\n🔄 Starting upscale process - {total_steps} tiles to process\n" + "="*50)

    def _send_websocket_progress(self):
        """Send progress via WebSocket to ComfyUI UI"""
        if not WEBSOCKET_PROGRESS_AVAILABLE or not self.node_id:
            return
            
        try:
            # Update progress state using the new API
            progress_state = get_progress_state()
            if progress_state:
                progress_state.update_progress(
                    self.node_id, 
                    self.current_sub_step, 
                    self.total_sub_steps
                )
            
            # Send progress text message
            if hasattr(PromptServer, 'instance') and PromptServer.instance:
                progress_text = f"Tile {self.current_step}/{self.total_steps} • Step {self.current_sub_step}/{self.total_sub_steps}"
                PromptServer.instance.send_progress_text(progress_text, self.node_id)
                
        except Exception as e:
            print(f"WebSocket progress failed: {e}")

    def update(self, sub_progress_step=None):
        if sub_progress_step is not None:
            # Update sub-progress within current tile
            self.current_sub_step = self.current_step * self.sub_steps_per_tile + sub_progress_step
        else:
            # Move to next tile
            self.current_step += 1
            self.current_sub_step = self.current_step * self.sub_steps_per_tile
        
        # Calculate percentages
        main_percentage = (self.current_step / self.total_steps) * 100
        sub_percentage = (self.current_sub_step / self.total_sub_steps) * 100
        
        # Console progress
        progress_bar = "█" * int(main_percentage // 5) + "░" * (20 - int(main_percentage // 5))
        print(f"\n🎯 Processing tile {self.current_step}/{self.total_steps} [{progress_bar}] {main_percentage:.1f}%")
        print(f"   📊 Overall progress: {self.current_sub_step}/{self.total_sub_steps} ({sub_percentage:.1f}%)\n")
        
        # WebSocket progress integration (preferred for new ComfyUI)
        self._send_websocket_progress()
        
        # Legacy progress integration (fallback for older ComfyUI)
        if LEGACY_PROGRESS_AVAILABLE and not WEBSOCKET_PROGRESS_AVAILABLE:
            try:
                comfy.utils.report_progress(self.current_sub_step, self.total_sub_steps)
            except Exception as e:
                print(f"Note: Legacy progress reporting failed: {e}")
        
        if self.current_step == self.total_steps:
            print("="*50 + f"\n✅ Upscale completed successfully! Processed {self.total_steps} tiles\n")

    def update_sub_progress(self, step_name, step_number):
        """Update sub-progress for detailed tracking within each tile"""
        self.current_sub_step = self.current_step * self.sub_steps_per_tile + step_number
        
        # Console sub-progress
        print(f"   ⚙️  {step_name} (Step {step_number}/{self.sub_steps_per_tile})")
        
        # WebSocket sub-progress integration
        if WEBSOCKET_PROGRESS_AVAILABLE and self.node_id:
            try:
                # Update progress state
                progress_state = get_progress_state()
                if progress_state:
                    progress_state.update_progress(
                        self.node_id, 
                        self.current_sub_step, 
                        self.total_sub_steps
                    )
                
                # Send detailed progress text
                if hasattr(PromptServer, 'instance') and PromptServer.instance:
                    detailed_text = f"Tile {self.current_step}/{self.total_steps} • {step_name} ({step_number}/{self.sub_steps_per_tile})"
                    PromptServer.instance.send_progress_text(detailed_text, self.node_id)
                    
            except Exception as e:
                pass  # Silent fallback for sub-progress
        
        # Legacy sub-progress integration (fallback)
        elif LEGACY_PROGRESS_AVAILABLE:
            try:
                comfy.utils.report_progress(self.current_sub_step, self.total_sub_steps)
            except Exception as e:
                pass  # Silent fallback for sub-progress

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
                "new_resolution": ("INT", {"default": 1072, "min": 16, "max": 16384, "step": 16}),
                "preserve_vram": ("BOOLEAN", {"default": False}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8}),
                "tile_upscale_resolution": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "tiling_strategy": (["Chess", "Linear"],),
            },
            "optional": {
                "block_swap_config": ("block_swap_config",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, seed, new_resolution, preserve_vram, tile_width, tile_height, mask_blur, tile_padding, tile_upscale_resolution, tiling_strategy, block_swap_config=None):
        try:
            # Get execution context for WebSocket progress
            context = None
            node_id = None
            
            if WEBSOCKET_PROGRESS_AVAILABLE:
                try:
                    context = get_executing_context()
                    if context:
                        node_id = context.node_id
                        
                        # Initialize progress state
                        progress_state = get_progress_state()
                        if progress_state:
                            progress_state.update_progress(node_id, 0, 100)
                        
                        # Send initial status
                        if hasattr(PromptServer, 'instance') and PromptServer.instance:
                            PromptServer.instance.send_progress_text("Initializing upscale process...", node_id)
                            
                except Exception as e:
                    print(f"Could not initialize WebSocket progress: {e}")
            
            # Setup
            seedvr2_instance = self._get_seedvr2_instance()
            pil_image = tensor_to_pil(image)
            upscale_factor = new_resolution / max(pil_image.width, pil_image.height)
            output_width = int(pil_image.width * upscale_factor)
            output_height = int(pil_image.height * upscale_factor)

            # Progress tracking
            main_tiles = self._generate_tiles(pil_image, tile_width, tile_height, tile_padding, tiling_strategy)
            progress = Progress(len(main_tiles))

            # Store original image for base creation
            self._original_image = pil_image
            
            # Main upscale pass with detail-preserving stitching
            output_image = self._process_and_stitch(main_tiles, output_width, output_height, seedvr2_instance, model, seed, tile_upscale_resolution, preserve_vram, block_swap_config, upscale_factor, mask_blur, progress)

            # Final progress report (completion)
            if WEBSOCKET_PROGRESS_AVAILABLE and node_id:
                try:
                    progress_state = get_progress_state()
                    if progress_state:
                        progress_state.update_progress(node_id, progress.total_sub_steps, progress.total_sub_steps)
                    
                    if hasattr(PromptServer, 'instance') and PromptServer.instance:
                        PromptServer.instance.send_progress_text("Upscale completed!", node_id)
                        
                except Exception as e:
                    print(f"Could not finalize WebSocket progress: {e}")
            
            # Legacy fallback
            elif LEGACY_PROGRESS_AVAILABLE:
                try:
                    comfy.utils.report_progress(progress.total_sub_steps, progress.total_sub_steps)
                except Exception:
                    pass

            return (pil_to_tensor(output_image),)
            
        except Exception as e:
            # Ensure progress is completed even on error
            if WEBSOCKET_PROGRESS_AVAILABLE and 'node_id' in locals() and node_id:
                try:
                    if 'progress' in locals():
                        progress_state = get_progress_state()
                        if progress_state:
                            progress_state.update_progress(node_id, progress.total_sub_steps, progress.total_sub_steps)
                except Exception:
                    pass
            elif LEGACY_PROGRESS_AVAILABLE:
                try:
                    if 'progress' in locals():
                        comfy.utils.report_progress(progress.total_sub_steps, progress.total_sub_steps)
                except Exception:
                    pass
            raise e

    def _get_seedvr2_instance(self):
        for node_class in nodes.NODE_CLASS_MAPPINGS.values():
            if node_class.__name__ == "SeedVR2":
                return node_class()
        raise RuntimeError("Could not find SeedVR2 node. Make sure it is installed correctly.")

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
            
            progress.update_sub_progress("Mask Blending", 3)
            
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
            
            # Complete this tile
            progress.update()

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

NODE_CLASS_MAPPINGS = {
    "UltimateSeedVR2Upscaler": UltimateSeedVR2Upscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSeedVR2Upscaler": "Ultimate SeedVR2 Upscaler"
}
