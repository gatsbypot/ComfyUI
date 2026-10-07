import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class CenterCropAndResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_and_resize"
    CATEGORY = "SpriteSheetTools"

    def crop_and_resize(self, image, target_size):
        B, H, W, C = image.shape
        target_aspect = 1.0
        current_aspect = W / H
        
        if current_aspect > target_aspect:
            scale = target_size / H
        else:
            scale = target_size / W
            
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        
        img_permuted = image.permute(0, 3, 1, 2)
        resized = F.interpolate(img_permuted, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        y_start = (new_h - target_size) // 2
        x_start = (new_w - target_size) // 2
            
        cropped = resized[:, :, y_start:y_start+target_size, x_start:x_start+target_size]
        final_images = cropped.permute(0, 2, 3, 1)
        
        return (final_images,)

class LumaKeyer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("RGBA_IMAGE", "ALPHA_PREVIEW")
    FUNCTION = "apply_luma_key"
    CATEGORY = "SpriteSheetTools"

    def apply_luma_key(self, image, threshold):
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
        
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        
        alpha = torch.where(luma < threshold, torch.zeros_like(luma), luma)
        alpha = alpha.unsqueeze(-1)
        
        rgba = torch.cat([image[..., :3], alpha], dim=-1)
        alpha_preview = torch.cat([alpha, alpha, alpha], dim=-1)
        
        return (rgba, alpha_preview)

class AIRemoveBackground:
    """
    Uses rembg to detect the object silhouette and remove the background.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "model": (["u2net", "isnet-general-use", "bria-rmbg", "u2netp", "silueta", "birefnet-general"], {"default": "bria-rmbg"}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("RGBA_IMAGE", "ALPHA_PREVIEW")
    FUNCTION = "remove_bg"
    CATEGORY = "SpriteSheetTools"

    def remove_bg(self, image, model):
        from rembg import remove, new_session
        import numpy as np
        from PIL import Image
        import torch
        
        session = new_session(model, providers=['CPUExecutionProvider'])
        
        out_frames = []
        out_alphas = []
        
        import comfy.utils
        pbar = comfy.utils.ProgressBar(image.shape[0])
        
        for i in range(image.shape[0]):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            if img_np.shape[-1] == 4:
                img_np = img_np[..., :3]
                
            pil_img = Image.fromarray(img_np)
            res_img = remove(pil_img, session=session)
            res_np = np.array(res_img).astype(np.float32) / 255.0
            
            out_frames.append(torch.from_numpy(res_np))
            
            alpha = res_np[..., 3:4]
            alpha_map = np.concatenate([alpha, alpha, alpha], axis=-1)
            out_alphas.append(torch.from_numpy(alpha_map))
            
            pbar.update_absolute(i + 1)
            
        return (torch.stack(out_frames), torch.stack(out_alphas))

class SpriteSheetFeatherMask:
    """
    Applies a Gaussian blur to the Alpha channel of an RGBA image.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "erode_amount": ("INT", {"default": 1, "min": 0, "max": 64, "step": 1}),
            "feather_amount": ("INT", {"default": 5, "min": 0, "max": 64, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("RGBA_IMAGE", "ALPHA_PREVIEW")
    FUNCTION = "apply_feather"
    CATEGORY = "SpriteSheetTools"

    def apply_feather(self, image, erode_amount, feather_amount):
        import scipy.ndimage
        import numpy as np
        import torch
        
        if feather_amount <= 0 and erode_amount <= 0:
            alpha = image[..., 3:4] if image.shape[-1] == 4 else torch.ones_like(image[..., :1])
            alpha_map = torch.cat([alpha, alpha, alpha], dim=-1)
            return (image, alpha_map)
            
        out_frames = []
        out_alphas = []
        
        for i in range(image.shape[0]):
            img_np = image[i].cpu().numpy()
            
            if img_np.shape[-1] != 4:
                out_frames.append(image[i])
                alpha = np.ones_like(img_np[..., :1])
                out_alphas.append(torch.from_numpy(np.concatenate([alpha, alpha, alpha], axis=-1)))
                continue
                
            original_alpha = img_np[..., 3:4]
            
            # Create a 2D boolean mask of the silhouette (preserve all non-zero alpha)
            silhouette_2d = (original_alpha[..., 0] > 0.0)
            
            # Calculate the exact pixel distance from the edge for every pixel inside the fire
            dist_inside = scipy.ndimage.distance_transform_edt(silhouette_2d)
            
            if feather_amount > 0:
                # Fade linearly from 0 (at the eroded boundary) to 1 (deep inside)
                factor = (dist_inside - erode_amount) / float(feather_amount)
            else:
                # Hard cut at the eroded boundary
                factor = dist_inside - erode_amount
                factor = np.where(factor > 0, 1.0, 0.0)
                
            # Clamp between 0 and 1
            factor = np.clip(factor, 0.0, 1.0).astype(np.float32)
            factor = np.expand_dims(factor, axis=-1)
            
            # Multiply original intricate alpha texture by this precise edge-only gradient
            final_alpha = original_alpha * factor
            
            img_np[..., 3:4] = final_alpha
            out_frames.append(torch.from_numpy(img_np))
            
            alpha_map = np.concatenate([final_alpha, final_alpha, final_alpha], axis=-1)
            out_alphas.append(torch.from_numpy(alpha_map))
            
        return (torch.stack(out_frames), torch.stack(out_alphas))

class SpriteSheetCompiler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "columns": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "cell_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "cell_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
            },
            "optional": {
                "rows": ("INT", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compile_sprite_sheet"
    CATEGORY = "SpriteSheetTools"

    def compile_sprite_sheet(self, images, columns, cell_width, cell_height, padding, rows=None):
        import math
        import torch
        import torch.nn.functional as F
        
        N, H, W, C = images.shape
        if N == 0:
            return (images,)
            
        if rows is None or rows <= 0:
            rows = math.ceil(N / columns)
        
        canvas_w = columns * cell_width + (columns + 1) * padding
        canvas_h = rows * cell_height + (rows + 1) * padding
        
        canvas = torch.zeros((1, canvas_h, canvas_w, C), dtype=images.dtype, device=images.device)
        
        for i in range(N):
            row = i // columns
            col = i % columns
            
            y_start = padding + row * (cell_height + padding)
            y_end = y_start + cell_height
            x_start = padding + col * (cell_width + padding)
            x_end = x_start + cell_width
            
            img = images[i].unsqueeze(0).permute(0, 3, 1, 2)
            if H != cell_height or W != cell_width:
                img = F.interpolate(img, size=(cell_height, cell_width), mode='bilinear', align_corners=False)
            img = img.permute(0, 2, 3, 1).squeeze(0)
            
            canvas[0, y_start:y_end, x_start:x_end, :] = img
            
        return (canvas,)





class LoadVideoSpriteSheet:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        import os
        input_dir = folder_paths.get_input_directory()
        try:
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        except Exception:
            files = []
        if not files:
            files = [""]
        return {
            "required": {
                "video": (files, {"video_upload": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("IMAGE", "ORIGINAL_FPS", "TOTAL_FRAMES")
    FUNCTION = "load_video"
    CATEGORY = "SpriteSheetTools"

    def load_video(self, video):
        import os
        import folder_paths
        video_path = os.path.join(folder_paths.get_input_directory(), video)
        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        import torchvision.io
        vframes, _, info = torchvision.io.read_video(video_path, pts_unit='sec')
        
        fps = info.get('video_fps', 30.0)
        total_frames = vframes.shape[0]
        
        if total_frames == 0:
            raise ValueError("Video contains no frames.")
            
        # Convert to ComfyUI format: float32, 0.0 to 1.0, shape (N, H, W, C)
        image_tensor = vframes.float() / 255.0
        
        return (image_tensor, float(fps), int(total_frames))

class VideoInfoAndConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "original_fps": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 120.0, "step": 0.1}),
                "target_fps": ("FLOAT", {"default": 15.0, "min": 0.1, "max": 120.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("IMAGE", "FPS", "FRAME_COUNT", "INFO_TEXT")
    FUNCTION = "convert"
    CATEGORY = "SpriteSheetTools"

    def convert(self, images, original_fps, target_fps):
        import torch
        total_frames = images.shape[0]
        if total_frames == 0:
            return (images, float(original_fps), int(total_frames), "No frames")
            
        step = original_fps / target_fps
        indices = torch.arange(0, total_frames, step=step).long()
        indices = torch.clamp(indices, 0, total_frames - 1)
        
        sampled_frames = images[indices]
        resulting_count = sampled_frames.shape[0]
        
        info_str = f"""Original FPS: {original_fps:.2f}
Original Frames: {total_frames}
Target FPS: {target_fps:.2f}
Resulting Frames: {resulting_count}"""
        
        return {"ui": {"text": [info_str]}, "result": (sampled_frames, float(target_fps), int(resulting_count), info_str)}

class VideoAnimatorPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "SpriteSheetTools"

    def preview(self, images, fps):
        import folder_paths
        import os
        import random
        import string
        import torchvision.io
        import torch

        if images.shape[0] == 0:
            return ()
            
        video_array = (images * 255.0).byte()
        
        output_dir = folder_paths.get_temp_directory()
        filename = f"video_preview_{''.join(random.choices(string.ascii_letters + string.digits, k=8))}.mp4"
        file_path = os.path.join(output_dir, filename)
        
        try:
            torchvision.io.write_video(file_path, video_array, fps=int(fps), video_codec='libx264')
        except Exception as e:
            print(f"Error saving video preview: {e}")
            return ()
        
        return {"ui": {"images": [{"filename": filename, "subfolder": "", "type": "temp", "format": "video/mp4"}]}}

class SpriteSheetAnimatorPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "columns": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "rows": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "total_frames": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "SpriteSheetTools"

    def preview(self, image, columns, rows, fps, total_frames):
        import folder_paths
        import os
        import random
        import string
        import torchvision.io
        import torch

        B, H_tot, W_tot, C = image.shape
        if B == 0 or total_frames <= 0:
            return ()
            
        cell_h = H_tot // rows
        cell_w = W_tot // columns
        
        frames = []
        for i in range(total_frames):
            row = i // columns
            col = i % columns
            if row >= rows:
                break
            y_start = row * cell_h
            y_end = y_start + cell_h
            x_start = col * cell_w
            x_end = x_start + cell_w
            frame = image[0, y_start:y_end, x_start:x_end, :]
            frames.append(frame)
            
        if len(frames) == 0:
            return ()
            
        video_tensor = torch.stack(frames, dim=0)
        video_array = (video_tensor * 255.0).byte()
        
        output_dir = folder_paths.get_temp_directory()
        filename = f"sprite_preview_{''.join(random.choices(string.ascii_letters + string.digits, k=8))}.mp4"
        file_path = os.path.join(output_dir, filename)
        
        try:
            torchvision.io.write_video(file_path, video_array, fps=int(fps), video_codec='libx264')
        except Exception as e:
            print(f"Error saving sprite animation preview: {e}")
            return ()
        
        return {"ui": {"images": [{"filename": filename, "subfolder": "", "type": "temp", "format": "video/mp4"}]}}
class VideoSpriteSheetFramer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_end": ("INT", {"default": 8, "min": 0, "max": 1000, "step": 1}),
                "layout_mode": (["Square", "Rectangle Compact"], {"default": "Square"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("IMAGE", "COLS", "ROWS", "SUGGESTION_TEXT")
    FUNCTION = "frame_video"
    CATEGORY = "SpriteSheetTools"

    def frame_video(self, images, trim_end, layout_mode):
        import math
        total_frames = images.shape[0]
        if total_frames == 0:
            return {"ui": {"text": ["No frames"]}, "result": (images, 0, 0, "No frames")}
            
        end_idx = max(0, total_frames - trim_end)
        trimmed_images = images[0:end_idx]
        N = trimmed_images.shape[0]
        
        # Calculate Square Layout
        sq_size = math.ceil(math.sqrt(N)) if N > 0 else 1
        sq_cols = sq_size
        sq_rows = sq_size
        sq_capacity = sq_cols * sq_rows
        sq_empty = sq_capacity - N
        
        # Calculate Rectangle Compact Layout
        best_capacity = float('inf')
        rect_cols = sq_cols
        rect_rows = sq_rows
        
        if N > 0:
            for c in range(1, N + 1):
                r = math.ceil(N / c)
                cap = c * r
                if cap >= N:
                    if cap < best_capacity:
                        best_capacity = cap
                        rect_cols = c
                        rect_rows = r
                    elif cap == best_capacity:
                        if abs(c - r) < abs(rect_cols - rect_rows):
                            rect_cols = c
                            rect_rows = r
        
        rect_empty = best_capacity - N

        suggestion = (f"Trimmed to {N} frames.\n"
                      f"Square Layout: {sq_cols}x{sq_rows} ({sq_empty} empty cells).\n"
                      f"Rectangle Compact: {rect_cols}x{rect_rows} ({rect_empty} empty cells).\n")
        
        if layout_mode == "Square":
            final_cols, final_rows = sq_cols, sq_rows
        else:
            final_cols, final_rows = rect_cols, rect_rows
            
        return {"ui": {"text": [suggestion]}, "result": (trimmed_images, final_cols, final_rows, suggestion)}

class VideoSeamlessLoopCrossfade:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "overlap_frames": ("INT", {"default": 5, "min": 1, "max": 64, "step": 1}),
                "split_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05}),
                "blend_curve": (["linear", "ease_in_out"], {"default": "linear"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "FRAME_COUNT")
    FUNCTION = "create_loop"
    CATEGORY = "SpriteSheetTools"

    def create_loop(self, images, overlap_frames, split_ratio, blend_curve):
        import torch
        import math
        
        N, H, W, C = images.shape
        k = overlap_frames
        
        if N <= k * 2:
            return (images, N)
            
        P = int(N * split_ratio)
        P = max(k, min(N - k, P))
        
        prefix = images[P : N - k]
        fade_out = images[N - k : N]
        fade_in = images[0 : k]
        suffix = images[k : P]
        
        t = torch.linspace(0.0, 1.0, steps=k, device=images.device)
        if blend_curve == "ease_in_out":
            weights = 0.5 - 0.5 * torch.cos(t * math.pi)
        else:
            weights = t
            
        weights = weights.view(k, 1, 1, 1)
        
        blended = (fade_out * (1.0 - weights)) + (fade_in * weights)
        final_sequence = torch.cat([prefix, blended, suffix], dim=0)
        
        return (final_sequence, final_sequence.shape[0])

class SpriteSheetAnimator:
    """
    Reverses a compiled sprite sheet back into a video batch for previewing.
    Uses the exact grid traversal logic from the Unity shader.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sprite_sheet": ("IMAGE",),
                "columns": ("INT", {"default": 14, "min": 1, "max": 64}),
                "rows": ("INT", {"default": 14, "min": 1, "max": 64}),
                "total_frames": ("INT", {"default": 184, "min": 0, "max": 4096}),
                # FPS is just a pass-through value to help configure your Video Combine node
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}), 
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("VIDEO_FRAMES", "FPS")
    FUNCTION = "unpack_to_video"
    CATEGORY = "SpriteSheetTools"

    def unpack_to_video(self, sprite_sheet, columns, rows, total_frames, fps):
        import torch
        
        # ComfyUI images are [Batch, Height, Width, Channels]
        # We assume sprite_sheet is a single image (Batch = 1)
        _, H, W, C = sprite_sheet.shape
        
        # Calculate the size of each individual frame in pixels
        cell_w = W // columns
        cell_h = H // rows
        
        # Shader clamping logic: protect against invalid frame counts
        grid_capacity = columns * rows
        frames_to_play = total_frames if total_frames > 0 else grid_capacity
        frames_to_play = min(frames_to_play, grid_capacity)
        
        video_frames = []
        
        # Replicate the shader's playback loop
        for i in range(frames_to_play):
            # 1. Find grid coordinates
            col_idx = i % columns
            row_idx = i // columns
            
            # 2. Convert to pixel coordinates
            x_start = col_idx * cell_w
            x_end = x_start + cell_w
            
            y_start = row_idx * cell_h
            y_end = y_start + cell_h
            
            # 3. Slice the tensor to extract the frame
            # tensor shape slicing: [batch, y_start:y_end, x_start:x_end, channels]
            frame = sprite_sheet[:, y_start:y_end, x_start:x_end, :]
            
            video_frames.append(frame)
            
        # 4. Concatenate all 1-frame batches into a single large video batch
        # This transforms a list of [1, H, W, C] into [frames_to_play, H, W, C]
        final_video = torch.cat(video_frames, dim=0)
        
        return (final_video, fps)

class SpriteSheetChannelExtractor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (["Red", "Green", "Blue", "Alpha", "Luma"], {"default": "Luma"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "extract_channel"
    CATEGORY = "SpriteSheetTools"

    def extract_channel(self, image, channel):
        import torch
        if channel == "Red":
            c = image[..., 0:1]
        elif channel == "Green":
            c = image[..., 1:2]
        elif channel == "Blue":
            c = image[..., 2:3]
        elif channel == "Alpha":
            if image.shape[-1] == 4:
                c = image[..., 3:4]
            else:
                c = torch.ones_like(image[..., 0:1])
        else: # Luma
            r = image[..., 0:1]
            g = image[..., 1:2]
            b = image[..., 2:3]
            c = 0.299 * r + 0.587 * g + 0.114 * b
            
        out = torch.cat([c, c, c], dim=-1)
        return (out,)

class SpriteSheetChannelPacker:
    """
    Packs separate grayscale images into the R, G, B, and optionally A channels 
    of a single output texture for optimized game engine materials.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "red_channel": ("IMAGE",),
                "green_channel": ("IMAGE",),
                "blue_channel": ("IMAGE",),
                "alpha_channel": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("PACKED_IMAGE",)
    FUNCTION = "pack_channels"
    CATEGORY = "SpriteSheetTools"

    def pack_channels(self, red_channel=None, green_channel=None, blue_channel=None, alpha_channel=None):
        import torch

        # Find any connected image to use as a shape template
        template = None
        for ch in (red_channel, green_channel, blue_channel, alpha_channel):
            if ch is not None:
                template = ch
                break
                
        if template is None:
            # If nothing is connected, just return a dummy 1x1 black image to avoid crashing
            return (torch.zeros((1, 1, 1, 3)),)

        def get_channel_data(ch, default_val=0.0):
            if ch is not None:
                return ch[..., 0:1]
            else:
                return torch.full((template.shape[0], template.shape[1], template.shape[2], 1), default_val, dtype=template.dtype, device=template.device)

        R = get_channel_data(red_channel, 0.0)
        G = get_channel_data(green_channel, 0.0)
        B = get_channel_data(blue_channel, 0.0)

        if alpha_channel is not None:
            A = get_channel_data(alpha_channel, 1.0)
            packed = torch.cat([R, G, B, A], dim=-1)
        else:
            packed = torch.cat([R, G, B], dim=-1)

        return (packed,)

class SpriteSheetShaderPreview:
    """
    Simulates the Unity HLSL shader math for color mapping and emission
    inside ComfyUI for rapid iteration and previewing.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fire_shape": ("IMAGE",),
                "emission_mask": ("IMAGE",),
                
                # A. Base Color Group
                "color_dark_hex": ("STRING", {"default": "#FF1A00"}), # Deep Red/Orange edges
                "color_light_hex": ("STRING", {"default": "#FFCC00"}), # Bright Yellow core
                
                # B. Emission Group
                "color_emission_hex": ("STRING", {"default": "#FFFFFF"}), # Bright White/Yellow glow
                "emission_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                
                # Output Format
                "transparent_background": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("PREVIEW_IMAGE",)
    FUNCTION = "simulate_shader"
    CATEGORY = "SpriteSheetTools"

    def hex_to_tensor(self, hex_str, device):
        import torch
        # Clean string and provide fallback if typed incorrectly
        hex_str = hex_str.strip().lstrip('#')
        if len(hex_str) != 6:
            hex_str = "FFFFFF"
        
        # Convert hex to 0.0-1.0 RGB float values
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        
        # Shape as [1, 1, 1, 3] to broadcast across the whole image tensor
        return torch.tensor([r, g, b], dtype=torch.float32, device=device).view(1, 1, 1, 3)

    def simulate_shader(self, fire_shape, emission_mask, color_dark_hex, color_light_hex, color_emission_hex, emission_strength, transparent_background):
        import torch
        
        device = fire_shape.device
        
        # 1. Extract raw grayscale channels
        shape_luma = fire_shape[..., 0:1]
        emission_luma = emission_mask[..., 0:1]
        
        # 2. Parse the UI hex colors into tensors
        color_dark = self.hex_to_tensor(color_dark_hex, device)
        color_light = self.hex_to_tensor(color_light_hex, device)
        color_emission = self.hex_to_tensor(color_emission_hex, device)
        
        # 3. HLSL Math Simulation
        
        # A. Base Color = lerp(color_dark, color_light, shape_luma)
        base_color = (color_dark * (1.0 - shape_luma)) + (color_light * shape_luma)
        
        # B. Final Emission = color_emission * emission_luma * emission_strength
        final_emission = color_emission * emission_luma * emission_strength
        
        # Final RGB = Base Color + Final Emission
        final_rgb = base_color + final_emission
        
        # Clamp to 0.0 - 1.0 range because ComfyUI canvas cannot render HDR bloom over 1.0
        final_rgb = torch.clamp(final_rgb, 0.0, 1.0)
        
        if transparent_background:
            # Append the fire_shape luma as the Alpha channel so the preview is transparent!
            final_preview = torch.cat([final_rgb, shape_luma], dim=-1)
        else:
            final_preview = final_rgb
        
        return (final_preview,)

NODE_CLASS_MAPPINGS = {
    "CenterCropAndResize": CenterCropAndResize,
    "LoadVideoSpriteSheet": LoadVideoSpriteSheet,
    "VideoInfoAndConverter": VideoInfoAndConverter,
    "VideoAnimatorPreview": VideoAnimatorPreview,
    "VideoSpriteSheetFramer": VideoSpriteSheetFramer,
    "VideoSeamlessLoopCrossfade": VideoSeamlessLoopCrossfade,
    "LumaKeyer": LumaKeyer,
    "AIRemoveBackground": AIRemoveBackground,
    "SpriteSheetFeatherMask": SpriteSheetFeatherMask,
    "SpriteSheetCompiler": SpriteSheetCompiler,
    "SpriteSheetAnimator": SpriteSheetAnimator,
    "SpriteSheetChannelExtractor": SpriteSheetChannelExtractor,
    "SpriteSheetChannelPacker": SpriteSheetChannelPacker,
    "SpriteSheetShaderPreview": SpriteSheetShaderPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CenterCropAndResize": "Center Crop and Resize",
    "LoadVideoSpriteSheet": "Load Video (SpriteSheetTools)",
    "VideoInfoAndConverter": "Video Info and Converter (SpriteSheetTools)",
    "VideoAnimatorPreview": "Video Animator Preview (SpriteSheetTools)",
    "VideoSpriteSheetFramer": "Video Sprite Sheet Framer (SpriteSheetTools)",
    "VideoSeamlessLoopCrossfade": "Seamless Loop Crossfader (SpriteSheetTools)",
    "LumaKeyer": "Luma Keyer",
    "AIRemoveBackground": "AI Remove Background (Rembg)",
    "SpriteSheetFeatherMask": "Sprite Sheet Feather Mask",
    "SpriteSheetCompiler": "Sprite Sheet Compiler (RGBA Safe)",
    "SpriteSheetAnimator": "Sprite Sheet Animator",
    "SpriteSheetChannelExtractor": "Channel Extractor (SpriteSheetTools)",
    "SpriteSheetChannelPacker": "Channel Packer (SpriteSheetTools)",
    "SpriteSheetShaderPreview": "Shader Preview (SpriteSheetTools)"
}
