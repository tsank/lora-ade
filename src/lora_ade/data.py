from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

DATASET = "ade-benchmark-corpus/ade_corpus_v2"
CONFIG = "Ade_corpus_v2_classification"
MODEL = "distilbert-base-uncased"

def load_splits(subset=None, seed=42):
    ds = load_dataset(DATASET, CONFIG)["train"]

    # drop duplicate sentences so none leaks between train and test
    seen = set()

    def keep(ex):
        if ex["text"] in seen:
            return False
        seen.add(ex["text"])
        return True

    ds = ds.filter(keep)
    if subset:
        ds = ds.train_test_split(train_size=subset, stratify_by_column="label", seed=seed)["train"]
    tmp = ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=seed)
    vt = tmp["test"].train_test_split(test_size=0.5, stratify_by_column="label", seed=seed)
    return tmp["train"], vt["train"], vt["test"]   # train, val, test = 80/10/10

def make_loaders(batch_size=32, subset=None, max_length=128):
    tok = AutoTokenizer.from_pretrained(MODEL)
    encode = lambda b: tok(b["text"], truncation=True, max_length=max_length)
    loaders = []
    for i, split in enumerate(load_splits(subset)):
        split = split.map(encode, batched=True, remove_columns=["text"])
        split = split.rename_column("label", "labels")
        loaders.append(DataLoader(split, batch_size=batch_size, shuffle=(i == 0),
                                  collate_fn=DataCollatorWithPadding(tok)))
    return loaders