# lora-ade

LoRA fine-tuning for adverse drug event (ADE) detection, built with PyTorch and Hugging Face PEFT, running locally on Apple Silicon.

Given a sentence from a medical case report, the model classifies it as **Related** (it describes an adverse reaction to a drug) or **Not-Related**. This is a basic pharmacovigilance task.

The project compares full fine-tuning of DistilBERT with LoRA, which trains about 1% of the weights.

## Results

Test set: 2,090 sentences. 3 epochs each. Single run, single seed.

| Mode | Trainable params | % of total | Train time (s) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| full | 66,955,010 | 100.00% | 1142 | 0.9278 | 0.8286 | 0.8150 | 0.8217 |
| lora | 739,586 | 1.09% | 775 | 0.9163 | 0.7593 | 0.8642 | 0.8083 |

- LoRA reaches F1 within about 1.3 points of full fine-tuning while training 1.09% of the weights, and takes about a third less time.
- LoRA finds more real adverse events (recall 0.86 against 0.82) but raises more false alarms (precision 0.76 against 0.83).
- The saved LoRA adapter is about 3 MB. A full model copy is about 268 MB.
- With one seed and a 2,090-sentence test set, the difference between the two modes is not conclusive.

Hardware: MacBook Air, Apple M4, 16 GB, using the PyTorch `mps` backend.

## How it works

- [docs/concepts.md](docs/concepts.md): plain-language explanation of the task, the model, LoRA and the code flow.
- [docs/architecture.md](docs/architecture.md): diagrams of the model, of LoRA inside one attention matrix, and of the code flow.

## Quick start

Requires Python 3.10 or newer (developed on 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

Train and compare (the first run downloads the dataset and DistilBERT, about 270 MB):

```bash
python -m lora_ade.train --mode full --epochs 3     # about 20 min on an M4 Air
python -m lora_ade.train --mode lora --epochs 3     # about 13 min; saves the adapter to outputs/lora_all
python -m lora_ade.compare                          # prints the comparison table
```

For a quick smoke test, add `--subset 2000 --epochs 1` to either training command.

Classify sentences with the saved adapter (run the `lora` training first):

```bash
python -m lora_ade.predict "The patient developed severe rash after starting ibuprofen."
```

Options for `train`: `--mode {full,lora}`, `--subset N`, `--epochs N`, `--batch-size N`, `--lr X`, `--r N`, `--alpha N`.

## Project layout

```
src/lora_ade/
    data.py         dataset download, deduplication, 80/10/10 split, tokenization
    train.py        model building (LoRA through PEFT), training loop, result saving
    evaluate.py     accuracy, precision, recall, F1
    compare.py      builds the results table from results/*.json
    predict.py      classifies sentences with the saved adapter
    lora_layer.py   hand-written LoRA layer for learning; not used by train.py
tests/              pytest checks for the hand-written LoRA layer
results/            JSON output of each run
docs/               concepts and architecture notes
```

## Setup details

- Base model: `distilbert-base-uncased`.
- LoRA: rank 8, alpha 16, dropout 0.1, applied to `q_lin` and `v_lin` in all 6 layers; the classifier head is also trained. Learning rate 5e-4 for LoRA and 5e-5 for full fine-tuning. These values are not tuned.
- Duplicate sentences are removed before splitting, so no sentence appears in both train and test.

## Data

ADE Corpus V2, classification configuration, loaded at run time from the Hugging Face Hub (`ade-benchmark-corpus/ade_corpus_v2`). The data catalog lists no known license, so the data is not included in this repository. Check the terms before reusing it.

Citation: Gurulingappa H., et al. (2012). Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports. *Journal of Biomedical Informatics*, 45(5), 885-892.

## Limitations

- Single run per setting with one seed.
- LoRA rank and learning rate are not tuned.
- Data comes from published case reports, not from real-world safety databases.
- This is a learning and portfolio project, not a validated clinical or regulatory tool.

## License

Code: MIT (see `LICENSE`). The dataset has its own terms, as described above.
