#!/bin/bash

# Clone llama.cpp and compile
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && git pull && LLAMA_CUBLAS=1 make
cd ..

# Install Python dependencies
pip install -q -r llama.cpp/requirements.txt
pip install -q huggingface_hub

# Clone the model repository (Placeholder for actual model download command)
git lfs install
git clone https://huggingface.co/google/gemma-7b

# Define the model ID and name for further processing
MODEL_ID="google/gemma-7b"
MODEL_NAME="gemma-7b"

# Convert the model weights to FP16 format
python llama.cpp/convert.py $MODEL_NAME --outtype f16 --outfile "${MODEL_NAME}/${MODEL_NAME}.fp16.bin"

# # Run the Python script to quantize and upload models
# python quantize_and_upload.py $MODEL_NAME
