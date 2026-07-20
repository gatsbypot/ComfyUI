# ComfyUI VFX Sprite Sheet Generator: Project Handover

This document summarizes the work completed to create a fully automated, seamless looping VFX generator in ComfyUI, specifically tailored for generating game-ready 2D sprite sheets (e.g., a Class C Electrical Fire).

## 1. Project Goal & Hardware Context
* **Objective:** Build a ComfyUI pipeline that takes a single reference image, uses it as the first and last frame of an animation to guarantee a perfect, seamless loop, and exports it as both a WebM video and an 8x8 game-ready sprite sheet.
* **Hardware:** The system operates on a Windows machine with an NVIDIA RTX 4070 (12GB VRAM). 
* **Model Choice:** Due to the 12GB VRAM constraint and the need for highly controllable motion, the **Stable Diffusion 1.5** architecture was selected alongside **AnimateDiff v3** (rather than SDXL).

## 2. Extensions & Models Installed
To achieve the exact keyframe injection required for perfect loops, we installed the following custom extensions and models:

* **[ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved):** The core engine for video generation.
* **[ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet):** Required to support SparseCtrl injection.
* **Models Downloaded:**
  * `v3_sd15_mm.ckpt`: The AnimateDiff v3 motion module.
  * `v3_sd15_sparsectrl_rgb.ckpt`: The SparseCtrl ControlNet model that reads RGB images and forces the motion module to match them at specific frames.

## 3. Custom Node Development
We developed a custom Python node to handle formatting requirements for the sprite sheet generator.

* **Node Name:** `VideoCropAndResize` (Added to `ComfyUI-SpriteSheetTools/nodes.py`)
* **Why:** The generation outputs raw frames that might not perfectly align with power-of-two dimensions. This node forces the video sequence into a strict 512x512 aspect ratio (using center cropping) *before* it gets passed to the `SpriteSheetGenerator`, ensuring perfectly uniform grid cells for the final game asset.

## 4. The Workflow Architecture (`animatediff_loop_workflow.json`)
We constructed a complete, drag-and-drop JSON workflow. Here is how the pipeline functions from start to finish:

1. **Input & Preprocessing:** 
   * A single image is loaded via `LoadImage`.
   * It is passed through the `ACN_SparseCtrlRGBPreprocessor` to properly scale and encode it for the SD 1.5 latent space.
2. **Keyframe Injection (SparseCtrl):**
   * The preprocessed image enters `ACN_AdvancedControlNetApply_v2`.
   * An `ACN_SparseCtrlIndexMethodNode` is attached with the index `0,-1`. This explicitly tells the AI: *"Make the 1st frame (0) and the last frame (-1) look exactly like the input image."* This guarantees a seamless loop.
3. **Motion Generation (AnimateDiff Gen2):**
   * The `ADE_UseEvolvedSampling` node wraps the base SD1.5 model.
   * We added an `ADE_StandardUniformContextOptions` node. **Why?** AnimateDiff v3 natively only supports up to 32 frames. Because we wanted a 64-frame output (for an 8x8 sprite sheet), we added this Context Window to smoothly slide a 16-frame window across all 64 frames, bypassing the model's hardcoded limit.
4. **Prompting:**
   * Text prompts were heavily optimized for VFX assets (e.g., `isolated on black, solid black background, VFX, game asset, 2D sprite frame`).
5. **Formatting & Output:**
   * `SaveWEBM`: Exports the raw video for previewing.
   * `VideoCropAndResize`: Crops the video to 512x512.
   * `BatchFrameSampler` & `SpriteSheetGenerator`: Compiles the 64 frames into a single 8x8 sprite sheet image grid.

## 5. Debugging & Iterations
During construction, we encountered and resolved several complex integration issues:
* **UI vs Internal Class Names:** The JSON initially used Python class names (like `AdvancedControlNetApply`) instead of the required ComfyUI `node_id` strings (like `ACN_AdvancedControlNetApply_v2`). The JSON was patched to reflect the exact UI schemas.
* **Gen1 vs Gen2 AnimateDiff:** Discovered that connecting `ADE_ApplyAnimateDiffModelSimple` directly to the `KSampler` throws an error in modern AnimateDiff setups. It must be routed through the `ADE_UseEvolvedSampling` wrapper first.
* **Missing Video Codecs:** Replaced a third-party `SaveVideo` node with ComfyUI's native `SaveWEBM` node to ensure compatibility without needing `VideoHelperSuite`.
* **SparseCtrl Preprocessor Error:** The `AdvancedControlNetApply` node threw an error because it received a raw image. We injected the `ACN_SparseCtrlRGBPreprocessor` to satisfy its input requirements.
* **64 Frame Latent Limit:** The generation crashed because AnimateDiff v3 has a 32-frame maximum. Fixed by injecting the `ADE_StandardUniformContextOptions` node.

## Conclusion
The workflow is now fully stable. The final `animatediff_loop_workflow.json` acts as a complete, self-contained pipeline. By simply loading the JSON, providing a starting image, and hitting "Queue Prompt", the system will handle SparseCtrl injection, context windowing, cropping, and sprite sheet generation automatically.
