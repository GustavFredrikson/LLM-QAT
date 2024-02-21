#!/bin/bash

# Ensure huggingface_hub CLI is installed
pip install -U "huggingface_hub[cli]"

# Step 1: Clone the repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Step 2: Compile the software
make

# Step 3: Download and extract the dataset
wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip -d data

# Adjusting the directory for the extracted files
# Make sure the file name matches what the perplexity tool expects
mv data/wikitext-2-raw-v1/wiki.test.raw data/

# Step 4: Download the model using Hugging Face CLI
# Specifying the correct model and adjusting cache-dir to place it correctly
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF --cache-dir ./models

# The model is expected to be in a specific path structure after download
# We need to ensure the model file name is correct and placed as expected
# Here we assume the model file is named correctly in the downloaded structure
# Find the downloaded model and move it to the expected location
model_path=$(find models -name "*.gguf")
echo "Model found at: $model_path"

# Ensure the target directory exists and move the model file there
mkdir -p models/7B
mv "$model_path" models/7B/ggml-model-q4_0.gguf

# Step 5: Run the perplexity calculation
# Ensure the file path to the dataset is correct
./perplexity -m models/7B/ggml-model-q4_0.gguf -f data/wiki.test.raw

# End of script
