import sys

import torch
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer

from lora_ade.data import MODEL

LABELS = ["Not-Related", "Related"]


def main():
    path = "outputs/lora_all"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoPeftModelForSequenceClassification.from_pretrained(path, num_labels=2)
    model = model.to(device).eval()
    for text in sys.argv[1:]:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            probs = model(**enc).logits.softmax(-1)[0]
        i = probs.argmax().item()
        print(f"{LABELS[i]} ({probs[i]:.2f}): {text}")


if __name__ == "__main__":
    main()