import subprocess
import os
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Define the model path and clone if it does not exist
repo_url = "https://github.com/ggerganov/llama.cpp"
repo_dir = Path("llama.cpp")
if not repo_dir.exists():
    subprocess.run(["git", "clone", repo_url], check=True)
os.chdir(repo_dir)

# Directory to store generated data
gen_data_dir = Path("gen_data")
gen_data_dir.mkdir(parents=True, exist_ok=True)

# Start index from command line argument, default to 0 if not provided
i_start = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Number of initial tokens for synthesizing data
n_vocab = 100

# File check and determination of inner and outer loop values
file_path = gen_data_dir / f"gen.chunk.{i_start:02}.jsonl"
if file_path.exists():
    with open(file_path, 'r') as file:
        lines = file.readlines()
    inner_loop = len(lines) % n_vocab
    outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

# Define and check the model path
model_path = Path("models/7B/llama-2-7b-chat.Q4_K_M.gguf")
model_dir = model_path.parent
model_dir.mkdir(parents=True, exist_ok=True)
if not model_path.exists():
    print("Model not found at", model_path, ", downloading...")
    url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"
    subprocess.run(["wget", "-O", str(model_path), url], check=True)

# Environment variable for llama.cpp
os.environ["LLAMA_CUBLAS"] = "1"
subprocess.run(["make", "-j"], check=True)

# Function to run llama.cpp with given parameters and return output
def run_llama(model_path, prompt, max_length, n_gpu_layers=100):
    result = subprocess.run(['./main', '-m', str(model_path), '-p', prompt, '-n', str(max_length), '--n-gpu-layers', str(n_gpu_layers)],
                            capture_output=True, text=True)
    return result.stdout

# Main process
def main(i_start=0):
    # Setup
    gen_data_dir = Path("gen_data")
    gen_data_dir.mkdir(parents=True, exist_ok=True)

    n_vocab = 100  # Number of initial tokens for synthesizing data
    file_path = gen_data_dir / f"gen.chunk.{i_start:02}.jsonl"

    if file_path.exists():
        with open(file_path, 'r') as file:
            lines = file.readlines()
        inner_loop = len(lines) % n_vocab
        outer_loop = len(lines) // n_vocab
    else:
        inner_loop = 0
        outer_loop = 0

    model_path = Path("models/7B/llama-2-7b-chat.Q4_K_M.gguf")

    total_iterations = (6 - (3 + outer_loop)) * ((i_start + 1) * n_vocab + inner_loop - (i_start * n_vocab + inner_loop))

    # Main loop to generate text

    for j in tqdm(range(3 + outer_loop, 6), desc='Outer loop'):
        for i in tqdm(range(i_start * n_vocab + inner_loop, (i_start + 1) * n_vocab), desc=f'Generating text for chunk {j}'):
            # Generate initial text with a random or predefined prompt
            initial_prompt = str(i)

            # First generation step with llama.cpp
            first_output = run_llama(model_path, initial_prompt, j)

            # Second generation step with llama.cpp, using the output of the first as the new prompt
            second_output = run_llama(model_path, first_output, 2048)

            # Save the final output to the JSONL file
            with open(file_path, 'a') as file:
                json.dump({"text": second_output}, file)
                file.write('\n')

if __name__ == "__main__":
    i_start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(i_start)