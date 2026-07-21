import torch
import torch.nn.functional as F

class VideoFrameResampler:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        import os
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        if not files:
            files = [""]
        return {
            "required": {
                "video": (files,),
                "target_fps": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 120.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("IMAGE", "fps", "total_frames")
    FUNCTION = "load_and_resample"
    CATEGORY = "SpriteSheetTools"

    def load_and_resample(self, video, target_fps):
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
            
        if target_fps <= 0:
            target_fps = fps
            
        # Sample frames based on fps ratio
        step = fps / target_fps
        indices = torch.arange(0, total_frames, step=step).long()
        
        # Ensure indices don't exceed total_frames - 1
        indices = torch.clamp(indices, 0, total_frames - 1)
        
        sampled_frames = vframes[indices]
        
        # Convert to ComfyUI format: float32, 0.0 to 1.0, shape (N, H, W, C)
        image_tensor = sampled_frames.float() / 255.0
        
        return (image_tensor, float(fps), int(total_frames))

class BatchFrameSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_frame_count": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample_frames"
    CATEGORY = "SpriteSheetTools"

    def sample_frames(self, images, target_frame_count):
        num_frames = images.shape[0]
        if num_frames == 0 or num_frames == target_frame_count:
            return (images,)
        
        indices = torch.linspace(0, num_frames - 1, target_frame_count).long()
        return (images[indices],)



class VideoCropAndResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "crop_mode": (["center", "top", "bottom", "left", "right"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_and_resize"
    CATEGORY = "SpriteSheetTools"

    def crop_and_resize(self, images, target_width, target_height, crop_mode):
        B, H, W, C = images.shape
        target_aspect = target_width / target_height
        current_aspect = W / H
        
        if current_aspect > target_aspect:
            scale = target_height / H
        else:
            scale = target_width / W
            
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        
        img_permuted = images.permute(0, 3, 1, 2)
        resized = F.interpolate(img_permuted, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        if crop_mode == "center":
            y_start = (new_h - target_height) // 2
            x_start = (new_w - target_width) // 2
        elif crop_mode == "top":
            y_start = 0
            x_start = (new_w - target_width) // 2
        elif crop_mode == "bottom":
            y_start = new_h - target_height
            x_start = (new_w - target_width) // 2
        elif crop_mode == "left":
            y_start = (new_h - target_height) // 2
            x_start = 0
        elif crop_mode == "right":
            y_start = (new_h - target_height) // 2
            x_start = new_w - target_width
            
        cropped = resized[:, :, y_start:y_start+target_height, x_start:x_start+target_width]
        final_images = cropped.permute(0, 2, 3, 1)
        
        return (final_images,)



class SpriteSheetFeatherMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "feather_amount": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("RGBA_IMAGE", "ALPHA_PREVIEW")
    FUNCTION = "feather_mask"
    CATEGORY = "SpriteSheetTools"

    def feather_mask(self, image, feather_amount):
        if image.shape[-1] != 4:
            # If not RGBA, just return as is (or could throw error)
            return (image, image)
            
        alpha = image[:, :, :, 3:4]
        alpha_permuted = alpha.permute(0, 3, 1, 2)
        
        if feather_amount > 0:
            kernel_size = feather_amount * 2 + 1
            alpha_permuted = F.avg_pool2d(alpha_permuted, kernel_size, stride=1, padding=feather_amount)
            
        new_alpha = alpha_permuted.permute(0, 2, 3, 1)
        
        new_image = image.clone()
        new_image[:, :, :, 3:4] = new_alpha
        
        alpha_preview = new_alpha.repeat(1, 1, 1, 3)
        
        return (new_image, alpha_preview)


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
            torchvision.io.write_video(file_path, video_array, fps=float(fps), video_codec='libx264')
        except Exception as e:
            print(f"Error saving sprite animation preview: {e}")
            return ()
        
        return {"ui": {"images": [{"filename": filename, "subfolder": "", "type": "temp", "format": "video/mp4"}]}}


NODE_CLASS_MAPPINGS = {
    "BatchFrameSampler": BatchFrameSampler,
    "VideoCropAndResize": VideoCropAndResize,
    "VideoFrameResampler": VideoFrameResampler,
    "SpriteSheetFeatherMask": SpriteSheetFeatherMask,
    "SpriteSheetCompiler": SpriteSheetCompiler,
    "SpriteSheetAnimatorPreview": SpriteSheetAnimatorPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchFrameSampler": "Batch Frame Sampler (SpriteSheetTools)",
    "VideoCropAndResize": "Video Crop And Resize (SpriteSheetTools)",
    "VideoFrameResampler": "Video Frame Resampler (SpriteSheetTools)",
    "SpriteSheetFeatherMask": "Sprite Sheet Feather Mask (SpriteSheetTools)",
    "SpriteSheetCompiler": "Sprite Sheet Compiler (SpriteSheetTools)",
    "SpriteSheetAnimatorPreview": "Sprite Sheet Animator Preview (SpriteSheetTools)"
}
