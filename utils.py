import re
from datasets import load_dataset
import pyarabic.araby as araby

DEFAULT_SYSTEM_PROMPT = '''
Below is a story idea. Write a short story based on this context.
'''.strip()
DEFAULT_ARABIC_SYSTEM_PROMPT = '''
Generate a poem based on the following title, and the given era:
'''.strip()


def clean_text(text):
    '''
    Cleans text from unnecessary characters.
    '''
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)

    return re.sub(r'\^[^ ]+', '', text)


def print_trainable_parameters(model):
    '''
    Prints the number of trainable parameters in the model.
    '''
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}'
    )


def generate_training_prompt(idea, story, system_prompt=DEFAULT_SYSTEM_PROMPT):
    '''
    Generates inputs for the model, which consist of:
    ### Instruction: {instructions for the model on what to do (generate a story)}

    ### Input: {idea of a story to generate}

    ### Response: {story from the dataset that suits the idea}.
    '''
    return f'''
### Instruction: {system_prompt}

### Input:
{idea.strip()}

### Response:
{story}
'''.strip()


def generate_text(data_point, field):
    '''
    Returns a dict with idea, story and input for the model.
    '''
    idea = clean_text(data_point['source_text'])
    story = clean_text(data_point['target_text'])

    return {
        'idea': idea,
        'story': story,
        field: generate_training_prompt(
            idea, story, system_prompt=DEFAULT_SYSTEM_PROMPT)}
     

def get_datasets(train_dataset_source_path, train_dataset_target_path,
                    val_dataset_source_path, val_dataset_target_path,
                    test_dataset_source_path, test_dataset_target_path,
                    field='prompt'):
    '''
    Returns train, validation and test datasets, in the format described in
    generate_text() and generate_training_prompt().
    '''
    train_dataset_source = load_dataset('text', data_files=train_dataset_source_path, split='train')
    train_dataset_target = load_dataset('text', data_files=train_dataset_target_path, split='train')
    val_dataset_source = load_dataset('text', data_files=val_dataset_source_path, split='train')
    val_dataset_target = load_dataset('text', data_files=val_dataset_target_path, split='train')
    test_dataset_source = load_dataset('text', data_files=test_dataset_source_path, split='train')
    test_dataset_target = load_dataset('text', data_files=test_dataset_target_path, split='train')

    # rename columns for merging
    train_dataset_source =  train_dataset_source.rename_column('text', 'source_text')
    train_dataset_target = train_dataset_target.rename_column('text', 'target_text')
    val_dataset_source = val_dataset_source.rename_column('text', 'source_text')
    val_dataset_target = val_dataset_target.rename_column('text', 'target_text')
    test_dataset_source = test_dataset_source.rename_column('text', 'source_text')
    test_dataset_target = test_dataset_target.rename_column('text', 'target_text')

    # merge datasets
    train_dataset = train_dataset_source.add_column('target_text', train_dataset_target['target_text']) 
    val_dataset = val_dataset_source.add_column('target_text', val_dataset_target['target_text']) 
    test_dataset = test_dataset_source.add_column('target_text', test_dataset_target['target_text'])

    # format datasets so that they contain fields: idea, story, $field
    train_dataset = train_dataset.map(generate_text, fn_kwargs={'field': field}, remove_columns=['source_text', 'target_text'])
    val_dataset = val_dataset.map(generate_text, fn_kwargs={'field': field}, remove_columns=['source_text', 'target_text'])
    test_dataset = test_dataset.map(generate_text, fn_kwargs={'field': field}, remove_columns=['source_text', 'target_text'])

    return train_dataset, val_dataset, test_dataset


def get_dataset(source_path, target_path, field='prompt'):
    '''
    Returns only one dataset.
    '''
    dataset_source = load_dataset('text', data_files=source_path, split='train')
    dataset_target = load_dataset('text', data_files=target_path, split='train')

    dataset_source =  dataset_source.rename_column('text', 'source_text')
    dataset_target = dataset_target.rename_column('text', 'target_text')

    dataset = dataset_source.add_column(
        'target_text', dataset_target['target_text']
    ) 

    dataset = dataset.map(generate_text, fn_kwargs={'field': field},
                            remove_columns=['source_text', 'target_text'])

    return dataset


def generate_arabic_training_prompt(example, field='poem'):
    '''
    Generates inputs for the model, which consist of:
    ### Instruction: {instructions for the model on what to do (generate a poem)}

    ### Input: {title of a poem + era}

    ### Response: {poem text}.
    '''
    verses = example['poem verses']
    era = example['poet era']
    title = example['poem title']
    
    if not era:
        example['poet era'] = 'العصر الجاهلي'
        era = example['poet era']
    if not title:
        example['poem title'] = 'العصر الجاهلي'
        title = example['poem title']
    
    poem = ''
    for verse in verses:
        verse = verse.strip()
        verse = araby.strip_diacritics(verse)
        poem += verse
        poem += '\n'

    example['poem'] = poem

    example[field] = f'''
### Instruction: {DEFAULT_ARABIC_SYSTEM_PROMPT}

### Input:
{title.strip()} , {era.strip()}

### Response:
{poem}
'''.strip()

    return example


def get_arabic_datasets(dataset_name='arbml/Ashaar_dataset', field='prompt'):
    '''
    Returns train, validation and test datasets for arabic, in the format described in
    generate_arabic_training_prompt().
    '''
    train_dataset = load_dataset(dataset_name,split='train[:80%]')
    val_dataset = load_dataset(dataset_name,split='train[80%:90%]')
    test_dataset = load_dataset(dataset_name,split='train[90%:]')

    train_dataset = train_dataset.map(generate_arabic_training_prompt, fn_kwargs={'field': field})
    val_dataset = val_dataset.map(generate_arabic_training_prompt, fn_kwargs={'field': field})
    test_dataset = test_dataset.map(generate_arabic_training_prompt, fn_kwargs={'field': field})

    train_dataset = train_dataset.select_columns([field, 'poem'])
    val_dataset = val_dataset.select_columns([field, 'poem'])
    test_dataset = test_dataset.select_columns([field, 'poem'])
    
    return train_dataset, val_dataset, test_dataset



