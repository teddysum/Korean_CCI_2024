
import argparse
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model


from src.data import CustomDataset, DataCollatorForSupervisedDataset


# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_id", type=str, required=True, help="model file path")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
g.add_argument("--epoch", type=int, default=5, help="training epoch")
g.add_argument("--lr", type=float, default=2e-5, help="learning rate")
g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
g.add_argument("--save_dir", type=str, default="resource/results", help="model save path")
# fmt: on

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'


def main(args):
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )
    peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"],
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                #modules_to_save=["embed_tokens","lm_head"]
                )
    model = AutoModelForCausalLM.from_pretrained(args.model_id ,
                                                 quantization_config=quantization_config)
    model = get_peft_model(model, peft_config)

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = CustomDataset("resource/data/대화맥락추론_train.json", tokenizer)
    # valid_dataset = CustomDataset("resource/data/대화맥락추론_dev.json", tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
        })
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    tr_args = TrainingArguments(
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=200,
        do_train=True,
        do_eval=False,
        optim="adamw_bnb_8bit",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.save_dir,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=tr_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    exit(main(parser.parse_args()))