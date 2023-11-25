# llm-optimized-fintuning

In this project, we are exploring the potential of the new efficient fine-tuning
method QLoRA, for Large Language Models (LLMs). We fine-tune three open-source
models: LLama2-7b and Mistral-7b English and Jais-13b Arabic. The fine-tuning
task for English is story generation. The task for the Arabic model is poem
generation.

## Datasets Description

One of the primary datasets we utilize is an English dataset "Hierarchical Neural Story Generation". This dataset contains a vase collection of 300K of human-written stories in the English language, which we can use to train our models and generate new stories. By analysing it, we assort several groups of tags ('Prompt Tags' Table). Most samples are related to 'Writing Prompt' ([ WP ]), so this fact makes it easier for model to study almost one kind of story.

**Prompt Tags:**

| Tags | % |
| --- | --- |
| `[ WP ]` Writing Prompt | 86.4
`[ IP ]` Image Prompt| 3.0
`[ CW ]` Constrained Writing |2.5
`[ EU ]` Established Universe |2.3
`[ OT ]` Off Topic|1.5
`[ Other ]` Other tags |1.2
`[ TT ]` Theme Thursday |1.1
`[ FF ]` Fast Finish |1.0
`[ PI ]` Prompt Inspired|0.4
`[ RF ]` Reality Fiction |0.4
`[ MP ]` Media Prompt|0.2
`[ PM ]` Prompt Me| 0.1 |

Additionally, we incorporate the poem Arabic dataset, known as Ashaar dataset, which we obtain from hugging face. This dataset contains a rich collection of 212K of Arabic poems that we can use to train our models and generate new poetry.


## Usage

For finetuning run:
```
bash finetune.sh
```
Feel free to adjust the parameters in your script to adapt the code for your PC and training regime.

For perplexity evaluation run:
```
bash ./evaluate/calculate_perplexity.sh
```
with respective checkpoint_path parameter.

For BERTScore evaluation run:
```
bash bert.sh
```
with respective checkpoint_path parameter.

## Checkpoints

Checkpoints with QLoRA adapters are available at:

- LLaMA2-7b 24k iterations: https://huggingface.co/boda/llama2-7b-story-generation-24k
- Mistral-7b 24k iterations: https://huggingface.co/boda/mistral-7b-story-generation-24k
- Jais-13b 170k iterations: https://huggingface.co/boda/jais-13b-poem-generation

Checkpoints of the models should be located at experiments/MODEL_NAME. Those checkpoints are for adapters only. Make sure, your main model(Llama or Mistral) is at ./home/username/.cache/huggingface/hub/... !
For Arabic, we use different function to load the dataset. Also for model_name and tokenizer, just pass the path to the Jais model.