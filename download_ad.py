from huggingface_hub import hf_hub_download
import shutil
import os

print("Downloading AnimateDiff v3 Motion Module...")
mm_path = hf_hub_download(repo_id="guoyww/animatediff", filename="v3_sd15_mm.ckpt")
shutil.copy(mm_path, r"d:\Abu\Tool\AI Workflow\comfyui\models\animatediff_models\v3_sd15_mm.ckpt")

print("Downloading SparseCtrl RGB...")
sparse_path = hf_hub_download(repo_id="guoyww/animatediff", filename="v3_sd15_sparsectrl_rgb.ckpt")
shutil.copy(sparse_path, r"d:\Abu\Tool\AI Workflow\comfyui\models\controlnet\v3_sd15_sparsectrl_rgb.ckpt")

print("Downloads complete!")
