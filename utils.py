import re
from datasets import load_dataset
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from tqdm import tqdm



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
    idea: str, story: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, train: bool = True
) -> str:
    

    ## For training, we need to provide the system prompt, the idea and the story
    if train:
        return f"""### Instruction: {system_prompt}

    ### Input:
    {idea.strip()}

    ### Response:
    {story}
""".strip()
    
    ## For validation and testing, we only need to provide the idea
    else:
        # print("in generating")
        # print(system_prompt)
        # print(f"""### Instruction: {system_prompt}

#     ### Input:
#     {idea.strip()}
# """.strip())
        return f"""### Instruction: {system_prompt}

### Input:
{idea.strip()}
### Response:
{story}
""".strip()


def generate_text(data_point, field, train):
    idea = clean_text(data_point["source_text"])
    story = clean_text(data_point["target_text"])

    return {
        "idea": idea,
        "story": story,
        field: generate_training_prompt(
            idea, story, system_prompt=DEFAULT_SYSTEM_PROMPT, train=train
        )
    }

def get_dataset(source_path, target_path, field='prompt', prompt_only=False):
    dataset_source = load_dataset('text', data_files=source_path, split="train")
    dataset_target = load_dataset('text', data_files=target_path, split="train")

    dataset_source =  dataset_source.rename_column("text", "source_text")
    dataset_target = dataset_target.rename_column("text", "target_text")

    dataset = dataset_source.add_column(
        "target_text", dataset_target['target_text']
    ) 

    dataset = dataset.map(
        generate_text, fn_kwargs={"field": field, "train": not prompt_only},
        remove_columns=["source_text", "target_text"], 
    )

    dataset.remove_columns(["idea"])

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


def compute_perplexity(model, tokenizer, predictions, batch_size, add_start_token=True, max_length=4096):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")
    print("len of encoded texts: ", len(encoded_texts))

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return np.sum(ppls)





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



