#!/bin/bash

# Ensure huggingface_hub CLI is installed
pip install -U "huggingface_hub[cli]"

# Step 1: Clone the repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp



# Step 3: Download and extract the dataset
wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip -o wikitext-2-raw-v1.zip -d data

# No need to move files since we will adjust the path in the perplexity command

# Step 4: Download the specific model file directly
mkdir -p models/7B
wget -O models/7B/llama-2-7b-chat.Q4_K_M.gguf "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"

# Step 5: Run the perplexity calculation
# Adjust the file path to the dataset correctly
make clean && LLAMA_CUBLAS=1 make -j

./perplexity -m models/7B/llama-2-7b-chat.Q4_K_M.gguf -f data/wikitext-2-raw/wiki.test.raw --n-gpu-layers 5

# End of script
