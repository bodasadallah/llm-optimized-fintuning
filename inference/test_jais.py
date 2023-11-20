import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig, StoppingCriteriaList, TextIteratorStreamer
from transformers import pipeline











adapters_name  = "experiments/jais/checkpoint-25000"
model_name = "/home/abdelrahman.sadallah/.cache/huggingface/hub/jais"
## Bits and Bytes config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    load_in_4bit=True,
    trust_remote_code=True,
    device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

model = PeftModel.from_pretrained(model, adapters_name)
model.eval()


def gen(model, text: str, tmp=0.1, top_p=0.6, max_length=1024):
    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_length,
                                do_sample=True,
                                top_p=top_p,
                                temperature=tmp)

    return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

base_text = '''### Instruction: Below is a prompt to generate a poem.

### Input:
'''

base_text = ''''''

# Happy kid was playing at the park, but then he broke his leg, and his life got completely changed.

with open('arabic_output.txt', 'a') as f:
    while 1:
        print('-'*20)
        # text = input("Enter a prompt: ")
        text = "أصبح الملك للذي فطر الخلق"
        print('-'*20)

        f.write('='*30 + '\n')
        f.write(text + '\n')
        f.write('-'*20 + '\n')

        text = base_text + text

        output = gen(model, text)

        output = output.replace('<newline>', '\n')

        f.write(output + '\n')
        f.write('='*30 + '\n')
        f.flush()

        print('-'*20)
        print(output)
        print('-'*20)
        break
