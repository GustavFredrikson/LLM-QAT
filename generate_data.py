# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os

model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
print("Loading tokenizer and model...")
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="gptq-4bit-64g-actorder_True")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
print("Model loaded!")

n_vocab = 100 # number of initial tokens for synthesizing data on each GPU.
i_start = sys.argv[1]
if os.path.exists("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl"):
    with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "r") as f:
        lines = f.readlines()
        inner_loop = len(lines) % n_vocab
        outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

if not os.path.exists("gen_data"):
    os.mkdir("gen_data")

for j in range(3 + outer_loop, 6):
    for i in range(int(i_start) * n_vocab + inner_loop, (int(i_start)+1) * n_vocab):
        print(i)
        input_ids = torch.tensor([[i]]).cuda()
        print(input_ids)
        # input_ids = torch.tensor([[i]])
        print("Generating original...")
        outputs1 = model.generate(input_ids, do_sample=False, max_length=j)
        print("Generating from outputs1...")
        outputs = model.generate(outputs1, do_sample=True, max_length=2048)
        print("Generated from outputs")
        gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        text_dict = {"text" : gen_text[0]}
        with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "a") as f:
            print("Saving...")
            f.write(json.dumps(text_dict))
            f.write('\n')