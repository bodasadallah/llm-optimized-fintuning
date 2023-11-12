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
    idea: str, story: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, treain: bool = True
) -> str:
    

    ## For training, we need to provide the system prompt, the idea and the story
    if treain:
        return f"""### Instruction: {system_prompt}

    ### Input:
    {idea.strip()}

    ### Response:
    {story}
""".strip()
    
    ## For validation and testing, we only need to provide the idea
    else:
        return f"""### Instruction: {system_prompt}

    ### Input:
    {idea.strip()}
""".strip()


def generate_text(data_point, field, train):
    idea = clean_text(data_point["source_text"])
    story = clean_text(data_point["target_text"])


    return {
        "idea": idea,
        "story": story,
        field: generate_training_prompt(idea, story, train),
    }

def get_dataset(source_path, target_path, field = 'prompt'):

    dataset_source = load_dataset('text',data_files=source_path, split="train")
    dataset_target = load_dataset('text',data_files=target_path, split="train")

    dataset_source =  dataset_source.rename_column("text", "source_text")
    dataset_target = dataset_target.rename_column("text", "target_text")

    dataset = dataset_source.add_column("target_text", dataset_target['target_text']) 

    dataset = dataset.map(generate_text, fn_kwargs={"field": field}, remove_columns=["source_text", "target_text"])

    return dataset
     

def get_datasets(train_dataset_source_path,
                  train_dataset_target_path,
                    val_dataset_source_path,
                      val_dataset_target_path,
                      test_dataset_source_path,
                        test_dataset_target_path,
                          field = 'prompt'):

    train_dataset_source = load_dataset('text',data_files=train_dataset_source_path, split="train")
    train_dataset_target = load_dataset('text',data_files=train_dataset_target_path, split="train")

    val_dataset_source = load_dataset('text',data_files=val_dataset_source_path,  split="train")
    val_dataset_target = load_dataset('text',data_files=val_dataset_target_path,  split="train" )

    test_dataset_source = load_dataset('text',data_files=test_dataset_source_path,  split="train")
    test_dataset_target = load_dataset('text',data_files=test_dataset_target_path,  split="train" )

    train_dataset_source =  train_dataset_source.rename_column("text", "source_text")
    train_dataset_target = train_dataset_target.rename_column("text", "target_text")
    val_dataset_source = val_dataset_source.rename_column("text", "source_text")
    val_dataset_target = val_dataset_target.rename_column("text", "target_text")
    test_dataset_source = test_dataset_source.rename_column("text", "source_text")
    test_dataset_target = test_dataset_target.rename_column("text", "target_text")

    train_dataset = train_dataset_source.add_column("target_text", train_dataset_target['target_text']) 
    val_dataset = val_dataset_source.add_column("target_text", val_dataset_target['target_text']) 
    test_dataset = test_dataset_source.add_column("target_text", test_dataset_target['target_text'])

    train_dataset = train_dataset.map(generate_text, fn_kwargs={"field": field, "train": True}, remove_columns=["source_text", "target_text"])
    val_dataset = val_dataset.map(generate_text, fn_kwargs={"field": field,"train": False}, remove_columns=["source_text", "target_text"])
    test_dataset = test_dataset.map(generate_text, fn_kwargs={"field": field,"train": False}, remove_columns=["source_text", "target_text"])

    return train_dataset, val_dataset, test_dataset




import pyarabic.araby as araby

DEFAULT_ARABIC_SYSTEM_PROMPT = """
Generate a poem based on the following title, and the given era:
""".strip()
def generate_arabic_training_prompt(
        
    example: dict,
    field: str = 'poem',
    train = True,
) -> str:
    

    verses = example['poem verses']
    era = example['poet era']
    title = example['poem title']
    
    if not era:
        example['poet era'] = "العصر الجاهلي"
        era = example['poet era']
    if not title:
        example['poem title'] = "العصر الجاهلي"
        title = example['poem title']
    
    

    poem = ''
    for verse in verses:

        verse = verse.strip()
        verse = araby.strip_diacritics(verse)
        poem += verse
        poem += '\n'


    example['poem'] = poem

    if train:
        example[field] = f"""### Instruction: {DEFAULT_ARABIC_SYSTEM_PROMPT}

    ### Input:
    {title.strip()} , {era.strip()}

    ### Response:
    {poem}
    """.strip()


    else:

        example[field] =  f"""### Instruction: {DEFAULT_ARABIC_SYSTEM_PROMPT}

    ### Input:
    {title.strip()} , {era.strip()}
    """.strip()  

    return example




def get_arabic_datasets(dataset_name = "arbml/Ashaar_dataset",field = 'prompt'):

    train_dataset = load_dataset(dataset_name,split='train[:80%]')
    val_dataset = load_dataset(dataset_name,split='train[80%:90%]')
    test_dataset = load_dataset(dataset_name,split='train[90%:]')




    train_dataset = train_dataset.map(generate_arabic_training_prompt, fn_kwargs={"field": field, "train" : True})
    val_dataset = val_dataset.map(generate_arabic_training_prompt, fn_kwargs={"field": field , "train" : False})
    test_dataset = test_dataset.map(generate_arabic_training_prompt, fn_kwargs={"field": field , "train" : False})


    train_dataset = train_dataset.select_columns([field, 'poem'])
    val_dataset = val_dataset.select_columns([field, 'poem'])
    test_dataset = test_dataset.select_columns([field, 'poem'])
    

    return train_dataset, val_dataset, test_dataset



