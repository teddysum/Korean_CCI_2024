
import argparse
import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data import CustomDataset


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output directory path to save artifacts")
g.add_argument("--model_id", type=str, required=True, help="model file path")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
g.add_argument("--device", type=str, required=True, help="the number of gpus")
# fmt: on


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # answer = [tokenizer.vocab['A'], tokenizer.vocab['B'], tokenizer.vocab['C']]
    dataset = CustomDataset("resource/data/대화맥락추론_test.json", tokenizer, args.device)

    result = []
    for idx in tqdm.tqdm(range(len(dataset))):
        inp, oup = dataset[idx]
        outputs = model(
            inp
        )
        logits = outputs.logits[:,-1].flatten()
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        result.append(str(probs))

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(result))


if __name__ == "__main__":
    exit(main(parser.parse_args()))