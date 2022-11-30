import torch
from transformers import AutoTokenizer

class dataset:
    def __init__(self, sentence, label, tokenizer) -> None:
        self.sentences = sentence
        self.labels  = label
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.sentences[idx]
        label = self.labels[idx]
        # let's tokenize
        tokenizedText = self.tokenizer(text,max_length = 512, padding="max_length", truncation=True, return_tensors="pt")
        # let's padd the tokens
        return idx, {
            "input_ids": tokenizedText["input_ids"].squeeze(),
            "attention_mask":  tokenizedText["attention_mask"].squeeze(),
            "labels": label
        }


    def __len__(self):
        return len(self.sentences)
