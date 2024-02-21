from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os

model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-64g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="gptq-4bit-64g-actorder_True")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
print("Loaded tokenizer and model")

# Wrap the generation in torch.no_grad() to reduce memory usage
with torch.no_grad():
    n_vocab = 500  # Adjusted for example, reduce if needed
    i_start = sys.argv[1]  # Ensure this is passed correctly
    if not os.path.exists("gen_data"):
        os.mkdir("gen_data")
    filepath = f"gen_data/gen.chunk.{str(i_start).zfill(2)}.jsonl"
    
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            lines = f.readlines()
            inner_loop = len(lines) % n_vocab
            outer_loop = len(lines) // n_vocab
    else:
        inner_loop = 0
        outer_loop = 0

    for j in range(3 + outer_loop, 6):
        for i in range(int(i_start) * n_vocab + inner_loop, (int(i_start) + 1) * n_vocab):
            print(i)
            input_ids = torch.tensor([[i]]).cuda()
            outputs1 = model.generate(input_ids, do_sample=False, max_length=j)
            outputs = model.generate(outputs1, do_sample=True, max_length=2048)
            gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            text_dict = {"text": gen_text[0]}
            with open(filepath, "a") as f:
                f.write(json.dumps(text_dict))
                f.write('\n')
