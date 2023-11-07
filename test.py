import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

adapters_name  = "experiments/checkpoint-8800"
model_name = "mistralai/Mistral-7B-v0.1"




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



def gen(model, text: str):
    inputs = tok(text, return_tensors="pt").to('cuda')
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256)
    return tok.decode(outputs[0][inputs_length:], skip_special_tokens=True)
     

base_text = '''### Instruction: Below is a story idea. Write a short story based on this context.


### Input:
'''

# Happy kid was playing at the park, but then he broke his leg, and his life got completely changed.


while 1:
    text = input("Enter a prompt: ")
    text = base_text.strip() + text

    print(gen(m, text))