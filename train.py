from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from transformers import TrainingArguments
from trl import SFTTrainer

dataset_source = load_dataset('text',data_files="datasets/english/writingPrompts/train.wp_source", split="train")
dataset_target = load_dataset('text',data_files="datasets/english/writingPrompts/train.wp_target", split="train")


dataset_source =  dataset_source.rename_column("text", "source_text")
dataset_target = dataset_target.rename_column("text", "target_text")

train_dataset = dataset_source.add_column("target_text", dataset_target['target_text']) 



import re
DEFAULT_SYSTEM_PROMPT = """
Below is a story idea. Write a short story based on this context.
""".strip()


def generate_training_prompt(
    idea: str, story: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    return f"""### Instruction: {system_prompt}

### Input:
{idea.strip()}

### Response:
{story}
""".strip()
     



def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\^[^ ]+", "", text)




def generate_text(data_point):
    idea = clean_text(data_point["source_text"])
    story = clean_text(data_point["target_text"])


    return {
        "idea": idea,
        "story": story,
        "prompt": generate_training_prompt(idea, story),
    }
     

train_dataset = train_dataset.map(generate_text, remove_columns=["source_text", "target_text"])


model_name = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
# model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

output_dir = "./experiments"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
)

from peft import LoraConfig

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64


lora_target_modules = [
    "q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj",
]

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules = lora_target_modules
    # target_modules=[
    #     "query_key_value",
    #     "dense",
    #     "dense_h_to_4h",
    #     "dense_4h_to_h",
    # ]
)


max_seq_length = 512


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="prompt",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)


for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)


trainer.train()

trainer.save_model()


print("Done training")
print(trainer.model)