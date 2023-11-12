import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

adapters_name  = "experiments/Llama-2-7b-hf/checkpoint-16000"
model_name = "meta-llama/Llama-2-7b-hf"


# adapters_name  = "experiments/Llama-2-7b-hf/checkpoint-16000"
# model_name = "mistralai/Mistral-7B-v0.1"





print(f"Starting to load the model {model_name} into memory")

m = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
m = PeftModel.from_pretrained(m, adapters_name)
# m = m.merge_and_unload()
tok = LlamaTokenizer.from_pretrained(model_name)
tok.bos_token_id = 1
stop_token_ids = [0]

print(f"Successfully loaded the model {model_name} into memory")



def gen(model, text: str, tmp=0.9, top_p=0.6, max_length=1024):
    inputs = tok(text, return_tensors="pt").to('cuda')
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_length,
                                do_sample=True,
                                top_p=top_p,
                                temperature=tmp)




    return tok.decode(outputs[0][inputs_length:], skip_special_tokens=True)
     

base_text = '''### Instruction: Below is a story idea. Write a short story based on this context.


### Input: a happy kid
'''

# Happy kid was playing at the park, but then he broke his leg, and his life got completely changed.

with open('generations.txt', 'a') as f:
    while 1:
        print('-'*20)
        text = input("Enter a prompt: ")
        print('-'*20)

        f.write('='*30 + '\n')
        f.write(text + '\n')
        f.write('-'*20 + '\n')

        text = base_text + text

        output = gen(m, text)

        output = output.replace('<newline>', '\n')

        f.write(output + '\n')
        f.write('='*30 + '\n')
        f.flush()

        print('-'*20)
        print(output)
        print('-'*20)
