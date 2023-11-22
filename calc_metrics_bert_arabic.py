import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from args_parser import get_args
from peft import PeftModel
from pathlib import Path
from utils import *
from tqdm import tqdm
from evaluate import load

tmp=0.9
top_p=0.6
max_length=300
# batch_size=18 
batch_size=16 # checkpoint

if __name__ == "__main__":

    bertscore = load("bertscore")
    args = get_args()
    model_name = args.model_name

    _, _, dataset = get_arabic_datasets(field = 'prompt') # Arabic
    dataset.cleanup_cache_files()

    print("1. Loaded dataset.")

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True
    # )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=bnb_config,
    #     trust_remote_code=True,
    #     device_map='auto'
    # )

    model = AutoModelForCausalLM.from_pretrained( # BASE
        model_name,
        trust_remote_code=True,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    model = PeftModel.from_pretrained(model, args.checkpoint_path) # checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.bos_token_id = 1
    tokenizer.pad_token = tokenizer.eos_token
    stop_token_ids = [0]

    print(f"2. Successfully loaded the model {model_name} into memory")


    output_dir = args.output_dir + f"/{model_name.split('/')[-1]}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving the model to {output_dir}")


    
    dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size
    )

    pres = 2631.791871905327
    reca =  2587.197250455618
    f1 = 2528.635782510042
    i = 0

    with torch.no_grad():
    # with torch.inference_mode():
         
        for batch in tqdm(dataloader): # ['idea', 'story', 'prompt']
            if i >= 271:


                inputs = tokenizer(batch["prompt"], padding=True, return_tensors="pt").to('cuda')
                
                
                outputs = model.generate(
                    **inputs, max_new_tokens=max_length, do_sample=True,
                    top_p=top_p, temperature=tmp,
                )

                generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated_texts = [res.split('Response:')[-1] for res in generated_texts]

                # print(len(generated_texts), len(batch['story']))
                # print(generated_texts)

                result = bertscore.compute(predictions=generated_texts, references=batch['poem'],
                                                    lang="ar") # precision, recall, F1

                pres += sum(result['precision'])
                reca += sum(result['recall'])
                f1 += sum(result['f1'])

                if i < 2:
                    # print(batch['prompt'], batch['poem'])
                    print(pres)
                    print(reca)
                    print(f1)
                    
                if i % 10 == 0:
                    # with open(args.output_dir + f"/{model_name.split('/')[-1]}" + '/bertscore_base.txt', 'a') as f:
                    with open(args.output_dir + f"/{model_name.split('/')[-1]}" + '/bertscore_90000.txt', 'a') as f:
                    # with open(args.output_dir + f"/{model_name.split('/')[-1]}" + '/bertscore_170000.txt', 'a') as f:
                        f.write(str(i+1) + ' ' + str(pres) + ' ' + str(reca) + ' ' + str(f1) + '\n')
                        f.flush()

                i += 1

            else:
                i += 1

    # with open(args.output_dir + f"/{model_name.split('/')[-1]}" + '/bertscore_base.txt', 'a') as f:
    with open(args.output_dir + f"/{model_name.split('/')[-1]}" + '/bertscore_90000.txt', 'a') as f:
    # with open(args.output_dir + f"/{model_name.split('/')[-1]}" + '/bertscore_170000.txt', 'a') as f:
        f.write(str(i) + ' ' + str(pres) + ' ' + str(reca) + ' ' + str(f1) + '\n')
        f.flush()
                 
    print("Average BERT Score: ", pres / (i*batch_size))

