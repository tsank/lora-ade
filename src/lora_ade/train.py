import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

from lora_ade.data import MODEL, make_loaders
from lora_ade.evaluate import evaluate

def count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "lora"], required=True)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lr = args.lr or (5e-4 if args.mode == "lora" else 5e-5)

    train_dl, val_dl, test_dl = make_loaders(args.batch_size, args.subset)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    if args.mode == "lora":
        cfg = LoraConfig(task_type="SEQ_CLS", r=args.r, lora_alpha=args.alpha,
                         lora_dropout=0.1, target_modules=["q_lin", "v_lin"])
        model = get_peft_model(model, cfg)   # also keeps the classifier head trainable
    model.to(device)

    trainable, total = count(model)
    print(f"mode={args.mode} device={device} lr={lr} trainable={trainable:,} of {total:,}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = evaluate(model, val_dl, device)
        print(f"epoch {epoch + 1}: val F1 {val['f1']:.4f} (last batch loss {loss.item():.4f})")
    train_time = time.time() - start

    test = evaluate(model, test_dl, device)
    print("TEST:", {k: round(v, 4) for k, v in test.items()}, f"| train time {train_time:.0f}s")

    if args.mode == "lora":
        Path("outputs").mkdir(exist_ok=True)
        model.save_pretrained(f"outputs/lora_{args.subset or 'all'}")
        
    Path("results").mkdir(exist_ok=True)
    out = {"mode": args.mode, "subset": args.subset, "epochs": args.epochs, "lr": lr,
           "trainable_params": trainable, "total_params": total,
           "train_seconds": round(train_time), **test}
    Path(f"results/{args.mode}_{args.subset or 'all'}.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
