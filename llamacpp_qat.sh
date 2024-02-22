#!/bin/bash

# Define the model path


# Directory to store generated data
GEN_DATA_DIR="gen_data"

# Check if GEN_DATA_DIR exists, if not create it
[ ! -d "$GEN_DATA_DIR" ] && mkdir -p "$GEN_DATA_DIR"

# Start index from command line argument, default to 0 if not provided
I_START=${1:-0}

# Number of initial tokens for synthesizing data
N_VOCAB=100

# File check and determination of inner and outer loop values
FILE="$GEN_DATA_DIR/gen.chunk.$(printf "%02d" $I_START).jsonl"
if [ -f "$FILE" ]; then
    LINES=$(cat $FILE | wc -l)
    INNER_LOOP=$(($LINES % $N_VOCAB))
    OUTER_LOOP=$(($LINES / $N_VOCAB))
else
    INNER_LOOP=0
    OUTER_LOOP=0
fi

mkdir -p models/7B
wget -O models/7B/llama-2-7b-chat.Q4_K_M.gguf "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"

MODEL_PATH="models/7B/llama-2-7b-chat.Q4_K_M.gguf"

# Step 5: Run the perplexity calculation
# Adjust the file path to the dataset correctly
LLAMA_CUBLAS=1 make -j

# Main loop to generate text
for (( j=3+OUTER_LOOP; j<6; j++ )); do
    for (( i=I_START*N_VOCAB+INNER_LOOP; i<(I_START+1)*N_VOCAB; i++ )); do
        echo "Processing index $i with max length $j"
        
        # Generate initial text with a random or predefined prompt
        INITIAL_PROMPT="$i"  # Placeholder for initial prompt generation logic
        
        # First generation step with llama.cpp
        ./main -m $MODEL_PATH -p "$INITIAL_PROMPT" -n $j > first_output.txt --n-gpu-layers 100
        
        # Read the generated output to use as a prompt for the second generation
        SECOND_PROMPT=$(<first_output.txt)
        
        # Second generation step with llama.cpp, using the output of the first as the new prompt
        ./main -m $MODEL_PATH -p "$SECOND_PROMPT" -n 2048 > second_output.txt --n-gpu-layers 100
        
        # Format and save the final output to the JSONL file
        echo "{\"text\": \"$(<second_output.txt)\"}" >> $FILE
    done
done

# Cleanup temporary files
rm -f first_output.txt second_output.txt
