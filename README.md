# llm-optimized-fintuning

In this project, we are exploring the potential of the new efficient fine-tuning
method QLoRA, for Large Language Models (LLMs). We fine-tune three open-source
models: LLama2-7b and Mistral-7b English and Jais-13b Arabic. The fine-tuning
task for English is story generation. The task for the Arabic model is poem
generation.

## Datasets Description

bla-bla-bla

## Usage

Finetuning is done by running the script finetune.sh. To get the perplexity of
the model run claculate_perplexity.sh with the checkpoint parameter.

Checkpoints of the models should be located at experiments/MODEL_NAME. Those
checkpoints are for adapters only. Make sure, your main model(Llama or Mistral)
is at ./home/username/.cache/huggingface/hub/... !

For Arabic, we use different function to load the dataset. Also for model_name
and tokenizer, just pass the path to the Jais model.

Prompt Tags:

    WP: Writing Prompt <!-- Main tag -->

    SP: Simple Prompt

    EU: Established Universe

    CW: Constrained Writing

    TT: Theme Thursday
    
    PM: Prompt Me
    
    MP: Media Prompt
    
    IP: Image Prompt
    
    PI: Prompt Inspired
    
    OT: Off Topic
    * OT as an Advertisement!
    
    RF: Reality Fiction
