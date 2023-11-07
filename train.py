from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from transformers import TrainingArguments
from trl import SFTTrainer
from evaluate import load
from peft import LoraConfig, prepare_model_for_kbit_training
from args_parser import get_args
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


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



def generate_text(data_point):
    idea = clean_text(data_point["source_text"])
    story = clean_text(data_point["target_text"])


    return {
        "idea": idea,
        "story": story,
        "prompt": generate_training_prompt(idea, story),
    }
     




if __name__ == "__main__":
    args = get_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))


    dataset_source = load_dataset('text',data_files="datasets/english/writingPrompts/train.wp_source", split="train")
    dataset_target = load_dataset('text',data_files="datasets/english/writingPrompts/train.wp_target", split="train")
    dataset_source =  dataset_source.rename_column("text", "source_text")
    dataset_target = dataset_target.rename_column("text", "target_text")
    train_dataset = dataset_source.add_column("target_text", dataset_target['target_text']) 
    train_dataset = train_dataset.map(generate_text, remove_columns=["source_text", "target_text"])


    model_name = args.model_name



    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )


    ## Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ## Prepare model for k-bit training
    # model = prepare_model_for_kbit_training(model)


    ## Print the number of trainable parameters
    print_trainable_parameters(model)

    ## Silence the warnings
    model.config.use_cache = False


    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'


    output_dir = args.output_dir

    per_device_train_batch_size =  args.per_device_train_batch_size
    per_device_val_batch_size = args.per_device_val_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps


    epoch_steps = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)


    optim = args.optim

    save_steps = args.save_steps
    logging_steps = args.logging_steps
    learning_rate = args.learning_rate
    max_grad_norm = args.max_grad_norm


    max_steps = epoch_steps * 10

    warmup_ratio = args.warmup_ratio
    lr_scheduler_type = args.lr_scheduler_type

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
        gradient_checkpointing=args.gradient_checkpointing,
    )


    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_r = args.lora_r


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


    max_seq_length = args.max_seq_length


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


    if args.checkpoint_path:
        trainer.train(args.checkpoint_path)
    else:
        trainer.train()


    trainer.save_model()


    print("Done training")
    print(trainer.model)
