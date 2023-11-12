import pathlib
# from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoTokenizer
# from transformers import TrainingArguments
# from trl import SFTTrainer
# from evaluate import load
# from peft import LoraConfig, prepare_model_for_kbit_training
from peft import PeftModel 
from args_parser import get_args
import re
from pathlib import Path
from utils import *
from tqdm import tqdm

import numpy as np
import evaluate

if __name__ == "__main__":
    args = get_args()

    # for arg in vars(args):
    #     print(arg, getattr(args, arg))



    # Get the datasets
    train_dataset, val_dataset, test_dataset = get_datasets(args.train_dataset_source_path, args.train_dataset_target_path,
                               args.val_dataset_source_path, args.val_dataset_target_path,
                               args.test_dataset_source_path, args.test_dataset_target_path,
                               field = args.field)

    val_dataset = val_dataset[:2]

    model_name = args.model_name

    print("1. Loaded datasets.")

    ## Bits and Bytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model = PeftModel.from_pretrained(model, args.checkpoint_path)
    # m = m.merge_and_unload()
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1
    stop_token_ids = [0]

    print(f"2. Successfully loaded the model {model_name} into memory")


    output_dir = args.output_dir


    output_dir = args.output_dir + f"/{model_name.split('/')[-1]}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving the model to {output_dir}")

    # for name, module in trainer.model.named_modules():
    #     if "norm" in name:
    #         module = module.to(torch.float32)


    # if args.checkpoint_path:
    #     trainer.train(args.checkpoint_path)
    # else:
    #     trainer.train()

    # model.load_state_dict(args.checkpoint_path)
    # print(val_dataset.keys())
    encodings = tokenizer("\n\n".join(val_dataset["prompt"][0]), return_tensors="pt")

    max_length = 4096
    stride = 512
    seq_len = encodings.input_ids.size(1)
    device = "cuda"

    print("before perplexity estimation.")

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids)
            
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        if begin_loc == 0:
            print("in the loop")

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print("Final perplexity is: ", ppl)
