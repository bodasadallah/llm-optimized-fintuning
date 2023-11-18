# from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoTokenizer
# from transformers import TrainingArguments
# from trl import SFTTrainer
# from evaluate import load
# from peft import LoraConfig, prepare_model_for_kbit_training
from peft import PeftModel 
from args_parser import get_args
from pathlib import Path
from utils import *
from tqdm import tqdm

# if you need to update datasets from cache, run once with next two lines uncommented
# from datasets import set_caching_enabled
# set_caching_enabled(False)

tmp=0.9
top_p=0.6
max_length=1024

if __name__ == "__main__":
    args = get_args()

    dataset = get_dataset(
        args.source_path, args.target_path, field="prompt", prompt_only=True
    )
    
    # dataset.cleanup_cache_files()

    model_name = args.model_name

    print("1. Loaded dataset.")

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
    tokenizer.pad_token = tokenizer.eos_token
    stop_token_ids = [0]

    print(f"2. Successfully loaded the model {model_name} into memory")


    output_dir = args.output_dir
    output_dir = args.output_dir + f"/{model_name.split('/')[-1]}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving the model to {output_dir}")

    dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.per_device_val_batch_size
    )

    pad_id  = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    loss_fct = CrossEntropyLoss(reduction="mean", ignore_index=pad_id)
    perplexities = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = tokenizer(batch["prompt"], padding=False, return_tensors="pt").to('cuda')

            story_lengths= len(tokenizer(batch["story"], padding=False, return_tensors="pt")['input_ids'][0])

            out_logits = model(**inputs).logits
            out_logits =  out_logits[0][-story_lengths:]
            labels_inputs = inputs['input_ids'][0][-story_lengths:]
            shifted_labels = labels_inputs[1:]
            shifted_logits = out_logits[:-1]

            perplexities.append(torch.exp(loss_fct(shifted_logits, shifted_labels)))

    print("Final perplexity is: ", torch.mean(perplexities))
    

    # loss_fct = CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token)
    # perplexities = []
    # with torch.no_grad:
    #     for batch in tqdm(dataloader):

    #         inputs = tokenizer(batch["prompt"], padding=True, return_tensors="pt").to('cuda')

    #         story_lengths= tokenizer(batch["story"], padding=False, return_tensors="pt")['in']
    #         inputs_length =  [len(i) for i in  inputs["input_ids"]]

    #         # with torch.inference_mode():
    #         #     outputs = model.generate(
    #         #         **inputs, max_new_tokens=max_length, do_sample=True,
    #         #         top_p=top_p, temperature=tmp
    #         #     )
    #         #     outputs = tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

    #         out_logits = model(**inputs).logits
    #         out_logits =  out_logits[:][inputs_length:]
    #         labels_inputs = inputs['input_ids'][:][inputs_length:]
    #         perplexity_batch = torch.exp(
    #         (loss_fct().sum(1)
    #         / shift_attention_mask_batch.sum(1))

    #     ppls += perplexity_batch.tolist()


    #         batch_perplexity = compute_perplexity(model, tokenizer, outputs, args.per_device_val_batch_size)
    #         perplexities.append(batch_perplexity)
    #         print("Batch perplexity is: ", batch_perplexity)

    #     ppl = np.mean(perplexities)
    #     print("Final perplexity is: ", ppl)
