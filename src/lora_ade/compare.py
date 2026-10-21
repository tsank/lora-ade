import json
from pathlib import Path

rows = [json.loads(p.read_text()) for p in sorted(Path("results").glob("*_all.json"))]
print("| Mode | Trainable params | % of total | Train time (s) | Accuracy | Precision | Recall | F1 |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    pct = 100 * r["trainable_params"] / r["total_params"]
    print(f"| {r['mode']} | {r['trainable_params']:,} | {pct:.2f}% | {r['train_seconds']} | "
          f"{r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")