# Concepts

Plain-language notes on what this project does and why. Diagrams are in [architecture.md](architecture.md).

## The task

Input: one sentence from a medical report.
Output: `Related` if the sentence describes an adverse reaction to a drug, `Not-Related` otherwise.
This is a basic pharmacovigilance task. Data: ADE Corpus V2 (Gurulingappa et al., 2012).

## The model

DistilBERT is a language model pre-trained on large amounts of English text, so it already understands sentences. It has about 67 million adjustable numbers (weights). It reads a sentence as follows:

1. The tokenizer cuts the text into pieces and converts each to a number.
2. Six identical transformer layers process the tokens. Each layer has self-attention (`q_lin`, `k_lin`, `v_lin`, `out_lin`) and a feed-forward network. Attention lets each word look at the other words to work out its meaning in context.
3. The vector of the first token serves as a summary of the sentence.
4. A small classifier head (`pre_classifier`, `classifier`) turns the summary into two scores.

## Fine-tuning

The pre-trained model knows English, not adverse drug events. Fine-tuning trains it on labelled sentences. Two ways:

- **Full fine-tuning:** update all 67 million weights. It works well, but it needs a full model copy (about 268 MB) for every task.
- **LoRA:** freeze every pre-trained weight and train small extra matrices beside selected layers, plus the new classifier head.

Analogy: the model is a printed textbook. Full fine-tuning rewrites the book. LoRA leaves the book as it is and adds sticky notes.

## How LoRA works

A weight matrix W of size 768 x 768 holds 589,824 numbers. LoRA keeps W frozen and learns a correction from two thin matrices:

- A: r x 768
- B: 768 x r (r is the rank, here 8)

The layer output becomes `y = W x + (alpha / r) * B (A x)`.

- A and B together have 2 x 8 x 768 = 12,288 numbers, about 2% of W.
- B starts at zero, so training begins from the original model's behaviour.
- `alpha / r` scales the correction. Here alpha = 16 and r = 8, so the scale is 2.
- After training, B x A can be added into W (`merge()`), so the adapter adds no inference cost.

In this project LoRA is applied to `q_lin` and `v_lin` in all 6 layers: 147,456 weights. The classifier head adds 592,130. Total trainable: 739,586 of 67,694,596 (1.09%).

## LoRA, PEFT and other methods

- **LoRA** is one technique.
- **PEFT** (parameter-efficient fine-tuning) is the family of techniques that train only a small part of a model. It is also the name of the Hugging Face `peft` library that implements them.
- `lora_layer.py` is a hand-written LoRA layer for learning. `train.py` uses the PEFT library, which finds the target layers in a real model, keeps the head trainable, and saves only the adapter.
- Other PEFT methods include AdaLoRA, DoRA, VeRA, IA3, LoHa/LoKr, OFT, prompt tuning, prefix tuning and P-tuning. QLoRA is LoRA on a 4-bit quantized base model; it is not used here because `bitsandbytes` does not run on Apple Silicon.

## Code flow

1. `data.py` downloads the dataset, removes duplicate sentences, splits it 80/10/10 into train, validation and test, and tokenizes it.
2. `train.py` loads DistilBERT, adds LoRA through PEFT (in `lora` mode), and runs the training loop for each batch: predict, compute loss, clear gradients, backward pass, update weights.
3. After each epoch it scores the validation set. After the last epoch it scores the test set once.
4. `evaluate.py` computes accuracy, precision, recall and F1, with "Related" as the positive class.
5. `compare.py` reads `results/*.json` and prints the comparison table.
6. `predict.py` loads the saved adapter and classifies sentences from the command line.

## Reading the metrics

- **Precision:** of the sentences flagged Related, the share that really are.
- **Recall:** of the truly Related sentences, the share that were found.
- **F1:** the balance of the two.
- In pharmacovigilance, a missed real event usually costs more than a false alarm, so recall matters.
- The splits have separate jobs: train to learn, validation to watch progress, test for the final honest score.

## Limits

- Single run per setting, single seed. Small differences between modes are not conclusive.
- LoRA settings (rank 8, learning rate 5e-4) are not tuned.
- The data is from published case reports and not from real-world safety databases.
