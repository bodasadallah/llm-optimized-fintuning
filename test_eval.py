# import pathlib
# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from args_parser import get_args
import re
from pathlib import Path
from utils import *
from calc_metrics import *



if __name__ == "__main__":
    args = get_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    output_dir = args.output_dir


    # Get the datasets
    test_dataset = get_datasets(test_dataset_source_path=args.test_dataset_source_path,
                                test_dataset_target_path=args.test_dataset_target_path,
                                field = args.field)

    print(f"Successfully loaded the test dataset")

    # Load the model in the checkpoint
    model_name = args.model_name
    checkpoint_name  = args.checkpoint_path # "experiments/Llama-2-7b-hf/checkpoint-16000"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type= args.bnb_4bit_quant_type # "nf4",
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype # torch.bfloat16,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant # True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        #trust_remote_code=True,
        use_flash_attention_2=args.use_flash_attention_2,
    )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     load_in_4bit=True,
    #     torch_dtype=torch.bfloat16,
    #     device_map='auto'
    # )

    model = PeftModel.from_pretrained(model, checkpoint_name)

    print(f"Successfully loaded the model {model_name} into memory")

    ## Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name) # LlamaTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1
    stop_token_ids = [0]
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'right'


    def gen(model, text: str, tmp=0.9, top_p=0.6, max_length=1024):
        inputs = tokenizer(text, return_tensors="pt").to('cuda')
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_length,
                                     do_sample=True,
                                     top_p=top_p,
                                     temperature=tmp)


        return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

    pred = []
    for story in test_dataset['story']:
        pred.extend(gen(model, text,
                        tmp=args.tmp, top_p=args.top_p, max_length=args.max_seq_length))

    print(f"Successfully generate all outputs")

    ppl = calc(test_dataset['idea'], pred) # FUNC FROM CALC_METRICS

    print('Final perplexity', ppl)
    # with open('test_metrics.txt', 'a') as f:
    #     f.write(ppl + '\n')
    #     f.flush()

