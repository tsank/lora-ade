# Architecture

Adverse drug event (ADE) detection with DistilBERT and LoRA.
Green nodes are trained; grey nodes are frozen.

## 1. Model

```mermaid
flowchart TD
    IN["Input sentence"] --> TOK["Tokenizer<br/>text to token IDs"]
    TOK --> EMB["Embeddings, 768-d"]
    EMB --> Q
    EMB --> K
    EMB --> V

    subgraph L["Transformer layer, repeated 6 times"]
        Q["q_lin 768x768<br/>+ LoRA A,B"]
        K["k_lin 768x768"]
        V["v_lin 768x768<br/>+ LoRA A,B"]
        O["out_lin 768x768"]
        FF["Feed-forward<br/>768 to 3072 to 768"]
        Q --> O
        K --> O
        V --> O
        O --> FF
    end

    FF --> CLS["First-token vector<br/>(sentence summary)"]
    CLS --> PRE["pre_classifier 768 to 768, ReLU"]
    PRE --> CLF["classifier 768 to 2"]
    CLF --> OUT["Not-Related / Related"]

    classDef frozen fill:#e0e0e0,stroke:#757575,color:#212121;
    classDef trained fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20;
    class EMB,K,O,FF frozen;
    class Q,V,PRE,CLF trained;
```

Trainable weights: 739,586 of 67,694,596 (1.09%).
LoRA adapters on q and v: 147,456. Classifier head: 592,130.

## 2. LoRA inside one attention matrix

```mermaid
flowchart LR
    X["input x"] --> W["W, frozen<br/>768x768"]
    X --> A["A, trainable<br/>768 to 8"]
    A --> B["B, trainable<br/>8 to 768"]
    B --> S["scale<br/>alpha / r = 2"]
    W --> ADD(("+"))
    S --> ADD
    ADD --> Y["output"]

    classDef frozen fill:#e0e0e0,stroke:#757575,color:#212121;
    classDef trained fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20;
    class W frozen;
    class A,B trained;
```

B starts at zero, so training begins from the original model's behaviour.

## 3. Code flow

```mermaid
flowchart TD
    HF1[("Hugging Face<br/>ADE dataset")] --> DATA["data.py<br/>dedupe, split 80/10/10,<br/>tokenize, batch"]
    HF2[("Hugging Face<br/>DistilBERT weights")] --> TRAIN
    DATA --> TRAIN["train.py<br/>build model, add LoRA via PEFT,<br/>training loop"]
    TRAIN --> EVAL["evaluate.py<br/>accuracy, precision,<br/>recall, F1"]
    TRAIN --> ADP["outputs/lora_all<br/>adapter, about 3 MB<br/>(not in git)"]
    EVAL --> RES["results/*.json"]
    RES --> CMP["compare.py<br/>comparison table"]
    ADP --> PRED["predict.py<br/>classify a sentence"]
```

`lora_layer.py` is a hand-written teaching version of LoRA and is not used by `train.py`.
