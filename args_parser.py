import argparse


def get_args():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    return args

def add_args(parser: argparse.ArgumentParser):

    parser.add_argument('--max_steps',
                            type=int,
                            default=10000)
    parser.add_argument('--save_steps',
                            type=int,
                            default=400)
    parser.add_argument('--logging_steps',
                            type=int,
                            default=100)
    parser.add_argument('--max_seq_length',
                            type=int,
                            default=512)
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default=None)
    parser.add_argument('--base_prompt',
                            type=str,
                            default="""Below is a story idea. Write a short story based on this context.""")
    
    parser.add_argument('--train_dataset_path',
                            type=str,
                            default='datasets/english/writingPrompts/train.wp_source')
    parser.add_argument('--val_dataset_path',
                            type=str,
                            default='datasets/english/writingPrompts/valid.wp_source')
    parser.add_argument('--model_name',
                            type=str,
                            default='meta-llama/Llama-2-7b-hf')

    parser.add_argument('--output_dir',
                                type=str,
                                default="./experiments")
    
    parser.add_argument('--per_device_train_batch_size',
                            type=int,
                            default=8)
    parser.add_argument('--per_device_val_batch_size',
                            type=int,
                            default=8)
    parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=2)
    parser.add_argument('--optim',
                            type=str,
                            default="paged_adamw_32bit")

    parser.add_argument('--learning_rate',
                            type=float,
                            default=2e-4)
    parser.add_argument('--max_grad_norm',
                            type=float,
                            default=0.3)

    parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.03)
    parser.add_argument('--lr_scheduler_type',
                            type=str,
                            default="constant")
    parser.add_argument('--group_by_length',
                            type=bool,
                            default=True)
    parser.add_argument('--gradient_checkpointing',
                            type=bool,
                            default=True)
    

    ##################################### LORA #####################################
    parser.add_argument('--lora_alpha',
                            type=int,
                            default=16)
    parser.add_argument('--lora_dropout',
                            type=float,
                            default=0.1)
    parser.add_argument('--lora_r',
                            type=int,
                            default=64)
    parser.add_argument('--bias',
                            type=str,
                            default="none")
    parser.add_argument('--task_type',
                            type=str,
                            default="CAUSAL_LM")
    parser.add_argument('--target_modules',
                            type=list,
                            default=[
                                "q_proj",
                                "up_proj",
                                "o_proj",
                                "k_proj",
                                "down_proj",
                                "gate_proj",
                                "v_proj",
                            ])
    ####################################################################################

