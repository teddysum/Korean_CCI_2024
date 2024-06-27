
import json
import pandas as pd

from datasets import Dataset
from tokenizers.processors import TemplateProcessing

class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer, device):
        self.inp = []
        self.oup = []

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat.append(f"화자{speaker}: {utterance}")

            question = f"[Question]\n위 대화의 {inp['category']}"
            if (ord(inp['category'][-1]) - ord("가")) % 28 > 0:
                question += "으로"
            else:
                question = "로"
            question += " 올바른 지문은?"
                
            chat = "\n".join(chat)
            chat = chat + "\n\n" + question + "\n\n[Option]\n"
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
            input_ids = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            self.inp.append(input_ids)
            self.oup.append(example["output"])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.oup[idx]
    