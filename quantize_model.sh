#!/bin/bash

# Define model and directories
MODEL_NAME="gemma-7b"
LLM_QAT_DIR="/root/LLM-QAT"
LLAMA_CPP_DIR="${LLM_QAT_DIR}/llama.cpp"

# Clone the LLM-QAT repository if it doesn't exist, or update it if it does
if [ ! -d "${LLM_QAT_DIR}" ]; then
    git clone https://github.com/GustavFredrikson/LLM-QAT.git ${LLM_QAT_DIR}
else
    (cd ${LLM_QAT_DIR} && git pull)
fi

# Clone and build llama.cpp
if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    git clone https://github.com/ggerganov/llama.cpp ${LLAMA_CPP_DIR}
    (cd ${LLAMA_CPP_DIR} && git pull && make clean && LLAMA_CUBLAS=1 make)
fi

# Install Python dependencies
pip install -q -r ${LLAMA_CPP_DIR}/requirements.txt
pip install -q huggingface_hub

echo "Setup and dependencies installed."
