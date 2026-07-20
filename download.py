from huggingface_hub import hf_hub_download
import shutil

print("Downloading SD1.5 safely...")
model_path = hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename="v1-5-pruned-emaonly.safetensors")
shutil.copy(model_path, r"d:\Abu\Tool\AI Workflow\comfyui\models\checkpoints\v1-5-pruned-emaonly.safetensors")
print("Download complete.")
