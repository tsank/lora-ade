import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch.pop("labels")
        preds += model(**batch).logits.argmax(-1).cpu().tolist()
        labels += y.cpu().tolist()
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f1}
