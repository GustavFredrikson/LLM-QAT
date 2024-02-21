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

# Step 4: Download the model using Hugging Face CLI
# Create the model directory
mkdir -p models/7B

# Use huggingface-cli to download the model
huggingface-cli download TheBloke/Llama-2-7b-Chat-GPTQ --cache-dir models/7B

# Assuming the model is downloaded to a directory, we might need to move/rename
# the model file to the expected name (ggml-model-q4_0.gguf). Adjust as needed.
# Placeholder command (adjust the source path as needed):
mv models/7B/TheBloke/Llama-2-7b-Chat-GPTQ/* models/7B/

# Step 5: Run the perplexity calculation
# Adjusting the file path to the extracted dataset
./perplexity -m models/7B/ggml-model-q4_0.gguf -f data/wiki.test.raw