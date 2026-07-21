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
            "feather_amount": ("INT", {"default": 5, "min": 0, "max": 64, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("RGBA_IMAGE", "ALPHA_PREVIEW")
    FUNCTION = "apply_feather"
    CATEGORY = "SpriteSheetTools"

    def apply_feather(self, image, feather_amount):
        import scipy.ndimage
        import numpy as np
        import torch
        
        if feather_amount <= 0:
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
                
            alpha = img_np[..., 3:4]
            sigma = feather_amount / 2.0
            alpha = scipy.ndimage.gaussian_filter(alpha, sigma=(sigma, sigma, 0))
            
            img_np[..., 3:4] = alpha
            out_frames.append(torch.from_numpy(img_np))
            
            alpha_map = np.concatenate([alpha, alpha, alpha], axis=-1)
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
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compile_sprite_sheet"
    CATEGORY = "SpriteSheetTools"

    def compile_sprite_sheet(self, images, columns, cell_width, cell_height, padding):
        import math
        import torch
        import torch.nn.functional as F
        
        N, H, W, C = images.shape
        if N == 0:
            return (images,)
            
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
class VideoFrameTrimmer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "trim_start": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "trim_end": ("INT", {"default": 5, "min": 0, "max": 1000, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "FRAME_COUNT")
    FUNCTION = "trim_frames"
    CATEGORY = "SpriteSheetTools"

    def trim_frames(self, images, trim_start, trim_end):
        total_frames = images.shape[0]
        if total_frames == 0:
            return (images, 0)
            
        start_idx = max(0, trim_start)
        end_idx = max(start_idx, total_frames - trim_end)
        
        trimmed_images = images[start_idx:end_idx]
        resulting_count = trimmed_images.shape[0]
        
        return (trimmed_images, int(resulting_count))

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

NODE_CLASS_MAPPINGS = {
    "CenterCropAndResize": CenterCropAndResize,
    "LoadVideoSpriteSheet": LoadVideoSpriteSheet,
    "VideoInfoAndConverter": VideoInfoAndConverter,
    "VideoAnimatorPreview": VideoAnimatorPreview,
    "VideoFrameTrimmer": VideoFrameTrimmer,
    "VideoSeamlessLoopCrossfade": VideoSeamlessLoopCrossfade,
    "LumaKeyer": LumaKeyer,
    "AIRemoveBackground": AIRemoveBackground,
    "SpriteSheetFeatherMask": SpriteSheetFeatherMask,
    "SpriteSheetCompiler": SpriteSheetCompiler,
    "SpriteSheetAnimatorPreview": SpriteSheetAnimatorPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CenterCropAndResize": "Center Crop and Resize",
    "LoadVideoSpriteSheet": "Load Video (SpriteSheetTools)",
    "VideoInfoAndConverter": "Video Info and Converter (SpriteSheetTools)",
    "VideoAnimatorPreview": "Video Animator Preview (SpriteSheetTools)",
    "VideoFrameTrimmer": "Video Frame Trimmer (SpriteSheetTools)",
    "VideoSeamlessLoopCrossfade": "Seamless Loop Crossfader (SpriteSheetTools)",
    "LumaKeyer": "Luma Keyer",
    "AIRemoveBackground": "AI Remove Background (Rembg)",
    "SpriteSheetFeatherMask": "Sprite Sheet Feather Mask",
    "SpriteSheetCompiler": "Sprite Sheet Compiler (RGBA Safe)",
    "SpriteSheetAnimatorPreview": "Sprite Sheet Animator Preview"
}
