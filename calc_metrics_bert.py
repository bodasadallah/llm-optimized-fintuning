# from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer # LlamaTokenizer, BitsAndBytesConfig, 
# from transformers import TrainingArguments
# from trl import SFTTrainer
# from evaluate import load
# from peft import LoraConfig, prepare_model_for_kbit_training
from peft import PeftModel 
from args_parser import get_args
from pathlib import Path
from utils import *
from tqdm import tqdm
from evaluate import load

# if you need to update datasets from cache, run once with next two lines uncommented
# from datasets import set_caching_enabled
# set_caching_enabled(False)

tmp=0.9
top_p=0.6
max_length=512

if __name__ == "__main__":

    bertscore = load("bertscore")
    args = get_args()
    
    for arg in vars(args):
        print(arg, getattr(args, arg))


    dataset = get_dataset(
        args.source_path, args.target_path, field="prompt", prompt_only=True
    )
    # dataset.cleanup_cache_files()

    model_name = args.model_name


    print("1. Loaded dataset.")


    ## Bits and Bytes config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type=args.bnb_4bit_quant_type,
    #     bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype, # torch.bfloat16,
    #     bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant, #True
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        # #trust_remote_code=True,
        # use_flash_attention_2=args.use_flash_attention_2,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model = PeftModel.from_pretrained(model, args.checkpoint_path)
    # m = m.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name) # LlamaTokenizer
    tokenizer.bos_token_id = 1
    tokenizer.pad_token = tokenizer.eos_token
    stop_token_ids = [0]

    print(f"2. Successfully loaded the model {model_name} into memory")


    output_dir = args.output_dir + f"/{model_name.split('/')[-1]}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving the model to {output_dir}")


    bert = []
    for i, source in tqdm(enumerate(dataset['idea'])):
            
        inputs = tokenizer(source, padding=True, return_tensors="pt").to('cuda') # padding=True,

        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=max_length, do_sample=True,
                top_p=top_p, temperature=tmp,
            )

        decode = tokenizer.decode(outputs[0], skip_special_tokens=True)

        res = decode.split('Response:')[-1]
        # res = res.replace('<newline> ', '\n')


        result = bertscore.compute(predictions=[res], references=[dataset['story'][i]],
                                            lang="en") # precision, recall, F1

        P, R, F1 = result['precision'], result['recall'], result['f1']
        
        if i < 4:
            print(len(res), len(dataset['story'][i]))
            print(source)
            print(res)
            print(P)
            print(R)
            print(F1)

        else:
            break
            
            
        


    print("BERT Score: ", bert)




    # dataset.set_format(type="torch")
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=args.per_device_val_batch_size
    # )

    # bert = []
    # i = 0
    # for batch in tqdm(dataloader):
    #     for source in batch["prompt"]:

    #         inputs = tokenizer(source, padding=True, return_tensors="pt").to('cuda') # padding=True,

    #         with torch.inference_mode():
    #             outputs = model.generate(
    #                 **inputs, max_new_tokens=max_length, do_sample=True,
    #                 top_p=top_p, temperature=tmp,
    #             )

    #         decode = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #         res = decode.split('Response:')[-1]
    #         res = res.replace('<newline> ', '\n')

    #         if i < 4:
    #             print(source)
    #             print(res)
                
    #             i+=1
    #         else:
    #             break
            



### To plot pairwise cosine similarity
# from bert_score import plot_example

# cand = cands[0]
# ref = refs[0]
# plot_example(cand, ref, lang="en")