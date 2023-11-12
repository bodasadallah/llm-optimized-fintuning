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

tmp=0.9
top_p=0.6
max_length=1024

if __name__ == "__main__":
    args = get_args()

    dataset = get_dataset(
        args.source_path, args.target_path, field="prompt", prompt_only=True
    )

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

    nlls = []
    for batch in tqdm(dataloader):
        inputs = tokenizer(batch["prompt"], padding=True, return_tensors="pt").to('cuda')

        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=max_length, do_sample=True,
                top_p=top_p, temperature=tmp
            )
        # tok.decode(outputs[0][inputs_length:], skip_special_tokens=True)
        with torch.no_grad():
            model_output = model(outputs, labels=outputs)

            neg_log_likelihood = model_output.loss

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean())
    print("Final perplexity is: ", ppl)

    # encodings = tokenizer("\n\n".join(dataset["prompt"]), return_tensors="pt")

    # max_length = 4096
    # stride = 512
    # seq_len = encodings.input_ids.size(1)
    # device = "cuda"

    # print("before perplexity estimation.")

    # nlls = []
    # prev_end_loc = 0
    # for begin_loc in tqdm(range(0, seq_len, stride)):
    #     end_loc = min(begin_loc + max_length, seq_len)
    #     trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    #     input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    #     target_ids = input_ids.clone()
    #     target_ids[:, :-trg_len] = -100

    #     with torch.no_grad():
    #         outputs = model(input_ids)
            
    #         # loss is calculated using CrossEntropyLoss which averages over valid labels
    #         # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
    #         # to the left by 1.
    #         neg_log_likelihood = outputs.loss

    #     nlls.append(neg_log_likelihood)

    #     if begin_loc == 0:
    #         print("in the loop")

    #     prev_end_loc = end_loc
    #     if end_loc == seq_len:
    #         break

    # ppl = torch.exp(torch.stack(nlls).mean())
    # print("Final perplexity is: ", ppl)
