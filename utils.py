import re
from datasets import load_dataset



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

def generate_text(data_point, field):
    idea = clean_text(data_point["source_text"])
    story = clean_text(data_point["target_text"])


    return {
        "idea": idea,
        "story": story,
        field: generate_training_prompt(idea, story),
    }

def get_dataset(source_path, target_path, field = 'prompt'):

    dataset_source = load_dataset('text',data_files=source_path, split="train")
    dataset_target = load_dataset('text',data_files=target_path, split="train")

    dataset_source =  dataset_source.rename_column("text", "source_text")
    dataset_target = dataset_target.rename_column("text", "target_text")

    dataset = dataset_source.add_column("target_text", dataset_target['target_text']) 

    dataset = dataset.map(generate_text, fn_kwargs={"field": field}, remove_columns=["source_text", "target_text"])

    return dataset
     

def get_datasets(train_dataset_source_path, train_dataset_target_path, val_dataset_source_path, val_dataset_target_path, field = 'prompt'):

    train_dataset_source = load_dataset('text',data_files=train_dataset_source_path, split="train")
    train_dataset_target = load_dataset('text',data_files=train_dataset_target_path, split="train")

    val_dataset_source = load_dataset('text',data_files=val_dataset_source_path,  split="train")
    val_dataset_target = load_dataset('text',data_files=val_dataset_target_path,  split="train" )


    train_dataset_source =  train_dataset_source.rename_column("text", "source_text")
    train_dataset_target = train_dataset_target.rename_column("text", "target_text")
    val_dataset_source = val_dataset_source.rename_column("text", "source_text")
    val_dataset_target = val_dataset_target.rename_column("text", "target_text")


    train_dataset = train_dataset_source.add_column("target_text", train_dataset_target['target_text']) 
    val_dataset = val_dataset_source.add_column("target_text", val_dataset_target['target_text']) 



    train_dataset = train_dataset.map(generate_text, fn_kwargs={"field": field}, remove_columns=["source_text", "target_text"])
    val_dataset = val_dataset.map(generate_text, fn_kwargs={"field": field}, remove_columns=["source_text", "target_text"])

    return train_dataset, val_dataset