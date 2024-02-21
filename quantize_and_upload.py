import os
import sys
from huggingface_hub import create_repo, HfApi

# Quantization methods and their details
quantization_methods = [
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", 
    "q4_0", "q4_k_s", "q4_k_m", "q5_0", 
    "q5_k_s", "q5_k_m", "q6_k", "q8_0"
]

model_name = sys.argv[1] if len(sys.argv) > 1 else "gemma-7b"
fp16_model_path = f"{model_name}/{model_name}.fp16.bin"

# Quantize the model
for method in quantization_methods:
    qtype = f"{model_name}/{model_name}.{method}.gguf"
    os.system(f"./llama.cpp/quantize {fp16_model_path} {qtype} {method}")

# Upload models to Hugging Face Hub
username = "YOUR_HUGGING_FACE_USERNAME"
token = "YOUR_HUGGING_FACE_TOKEN"

# Create a repo on Hugging Face Hub
repo_name = f"{username}/{model_name}-GGUF"
create_repo(repo_id=repo_name, repo_type="model", exist_ok=True, token=token)

api = HfApi()

# Upload quantized models
for method in quantization_methods:
    file_path = f"{model_name}/{model_name}.{method}.gguf"
    if os.path.exists(file_path):
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"{method}.gguf",
            repo_id=repo_name,
            token=token
        )

print("Quantization and upload completed.")
