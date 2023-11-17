import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel#, prepare_model_for_int8_training #get_peft_config, prepare_model_for_training
from args_parser import get_args
from pathlib import Path
from utils import *
from tqdm import tqdm
from evaluate import load
# from huggingface_hub import hf_hub_download
# import itertools

tmp=0.9
top_p=0.6
max_length=512
batch_size=64 #16

if __name__ == "__main__":

    bertscore = load("bertscore")
    args = get_args()
    model_name = args.model_name

    dataset = get_dataset(
        args.source_path, args.target_path, field="prompt", prompt_only=True
    )
    dataset.cleanup_cache_files()


    print("1. Loaded dataset.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # lora_alpha = args.lora_alpha
    # lora_dropout = args.lora_dropout
    # lora_r = args.lora_r
    # lora_target_modules = args.lora_target_modules

    # peft_config = LoraConfig(
    #     lora_alpha=lora_alpha,
    #     lora_dropout=lora_dropout,
    #     r=lora_r,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules = lora_target_modules
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map='auto'
    )

    # model = prepare_model_for_int8_training(model)
    # model = prepare_model_for_training(model)

    # peft_config = get_peft_config(config)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     load_in_4bit=True,
    #     torch_dtype=torch.bfloat16,
    #     device_map='auto'
    # )
    # model = PeftModel.from_pretrained(model, args.checkpoint_path)
    
    # config_file = hf_hub_download(model_name, 'adapter_config.json')

    model_tuple = PeftModel.from_pretrained(model, model_id='/home/anastasiia.demidova/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf'), #config=peft_config) # Baseline
    # model = PeftModel.from_pretrained(model, model_id='/home/anastasiia.demidova/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1'), # Baseline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1
    tokenizer.pad_token = tokenizer.eos_token
    stop_token_ids = [0]

    # print(model)
    
    # model = model_tuple[0]
    # model = tf.convert_to_tensor(model)
    model.eval()

    print(f"2. Successfully loaded the model {model_name} into memory")


    output_dir = args.output_dir + f"/{model_name.split('/')[-1]}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving the model to {output_dir}")


    
    dataset.set_format(type="torch")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size
    )

    # # Skip N batchs
    # for _ in itertools.islice(dataloader, 530): ### ??????????????
    #     pass
 
    pres = 0
    reca = 0
    f1 = 0
    i = 0
    for batch in tqdm(dataloader): # ['idea', 'story', 'prompt']
        
        # if i >= 161:

        inputs = tokenizer(batch["prompt"], padding=True, return_tensors="pt").to('cuda')
        # inputs = tokenizer(batch["prompt"], padding=True, return_tensors="pt").input_ids

        # with torch.no_grad():
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=max_length, do_sample=True,
                # input_ids=inputs.to("cuda"), max_new_tokens=max_length, do_sample=True,
                top_p=top_p, temperature=tmp,
            )

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = [res.split('Response:')[-1] for res in generated_texts]

        # print(len(generated_texts), len(batch['story']))
        # print(generated_texts)

        result = bertscore.compute(predictions=generated_texts, references=batch['story'],
                                            lang="en") # precision, recall, F1

        pres += sum(result['precision'])
        reca += sum(result['recall'])
        f1 += sum(result['f1'])

        if i < 2:
            print(pres)
            print(reca)
            print(f1)
            
        if i % 10 == 0:
            with open(args.output_dir + f"/{model_name.split('/')[-1]}" + '/bertscore.txt', 'a') as f:
                f.write(str(i+1) + ' ' + str(pres) + ' ' + str(reca) + ' ' + str(f1) + '\n')
                f.flush()

        i += 1

        # else:
        #     i += 1

    with open(args.output_dir + f"/{model_name.split('/')[-1]}" + '/bertscore.txt', 'a') as f:
                f.write(str(i) + ' ' + str(pres) + ' ' + str(reca) + ' ' + str(f1) + '\n')
                f.flush()
                
    print("Average BERT Score: ", pres / (i*batch_size))

