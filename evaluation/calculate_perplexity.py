from pathlib import Path

import torch
from peft import PeftModel
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          AutoTokenizer)

from args_parser import get_args
from utils import *

# if you need to update datasets from cache, run once with next two lines uncommented
# from datasets import set_caching_enabled
# set_caching_enabled(False)

if __name__ == '__main__':
    args = get_args()
    print(args)

    # dataset = get_dataset(
    #     args.source_path, args.target_path, field='prompt'
    # )
    _, _, dataset = get_arabic_datasets()

    model_name = args.model_name

    print('1. Loaded dataset.')

    # Bits and Bytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',  
        # bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Combine model with BnB
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map='auto'
    )
    # load the model from a checkpoint. Comment to get the not fine-tuned model
    model = PeftModel.from_pretrained(model, args.checkpoint_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.bos_token_id = 1
    tokenizer.pad_token = tokenizer.eos_token
    stop_token_ids = [0]

    print(f'2. Successfully loaded the model {model_name} into memory')


    output_dir = args.output_dir
    output_dir = args.output_dir + f'/{model_name.split("/")[-1]}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f'Saving the model to {output_dir}')

    dataset.set_format(type='torch')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.per_device_val_batch_size
    )

    pad_id  = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    loss_fct = CrossEntropyLoss(reduction='mean', ignore_index=pad_id)
    perplexities = []
    nan_count = 0

    # IMPORTANT: currently works for batch_size=1 (or for the first element of the batch)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = tokenizer(
                batch['prompt'], padding=False, return_tensors='pt', truncation=True, max_length=2048).to('cuda')

            story_length= len(tokenizer(batch['poem'], padding=False, return_tensors='pt')['input_ids'][0])

            out_logits = model(**inputs).logits
            # cut out the instruction
            out_logits =  out_logits[0][-story_length:]
            labels_inputs = inputs['input_ids'][0][-story_length:]
            # shift labels to the left by one
            shifted_labels = labels_inputs[1:]
            shifted_logits = out_logits[:-1]
            
            ppl = torch.exp(loss_fct(shifted_logits, shifted_labels))
            if not ppl.isnan():
                perplexities.append(ppl)
                nan_count += 1
            # print(ppl, end="\n")
            

    print('Final perplexity is: ', torch.mean(torch.tensor(perplexities)), nan_count)
    
