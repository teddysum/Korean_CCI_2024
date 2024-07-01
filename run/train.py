
import argparse

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaForSequenceClassification

from src.data import CustomDataset


# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_id", type=str, required=True, help="model file path")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--epoch", type=int, default=5, help="training epoch")
g.add_argument("--lr", type=float, default=2e-5, help="learning rate")
g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
g.add_argument("--save_dir", type=str, default="resource/model", help="model save path")
# fmt: on


def main(args):
    # initial model, optimizer, dataloader and acclerator
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.train()

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.padding_side = "left"

    train_dataset = CustomDataset("resource/data/대화맥락추론_train.json", tokenizer)
    # valid_dataset = CustomDataset("resource/data/대화맥락추론_dev.json", tokenizer)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
    )

    CELoss = CrossEntropyLoss()
    def loss_fn(targets, outputs):
        targets = torch.tensor([targets+tokenizer.vocab['A']]).to(args.device)
        logits = outputs.logits[:,-1]
        loss = CELoss(logits, targets)

        return loss

    step = 0
    for ep in range(args.epoch):
        for inp, trg in train_dataset:
            outputs = model(
                inp.to(args.device)
            )
            loss = loss_fn(trg, outputs) / args.gradient_accumulation_steps
            loss.backward()
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                print(f"Loss: {loss}, Step: {step}")
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
        print(f"Epoch: {ep+1}")
        model.save_pretrained(args.save_dir+f"/{ep+1}")
        

if __name__ == "__main__":
    exit(main(parser.parse_args()))