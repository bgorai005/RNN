# Bengali Transliteration without Attention Mechanism

This project implements a character-level Seq2Seq model for transliterating words from Latin script (English) to Bengali script using the Dakshina dataset. The model is trained **without an attention mechanism** and optimized using a hyperparameter sweep with **Weights & Biases (Wandb)**.

---

##  Overview

- **Dataset**: Dakshina (Bengali)
- **Task**: Transliteration (Latin → Bengali)
- **Model**: Encoder–Decoder (RNN/GRU/LSTM)
- **Framework**: PyTorch
- **Evaluation**: Sequence accuracy on test set
- **Hardware**: Kaggle with GPU (if available)

---

##  Project Structure

```
without_attention

---

##  Dataset

**Path on Kaggle**:  
`/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/`

| File | Purpose |
|------|---------|
| `bn.translit.sampled.train.tsv` | Training set |
| `bn.translit.sampled.dev.tsv`   | Validation set |
| `bn.translit.sampled.test.tsv`  | Test set |

- Each file is tab-separated with two columns:
  - Column 1: Target word (Bengali)
  - Column 2: Source word (Latin)

---

##  Model Architecture

### 🔹 Encoder
- Input: Latin character sequence
- Embedding layer: `embedding_dim`
- RNN: `RNN`, `GRU`, or `LSTM` with `hidden_size` and `num_layers`
- Dropout: Applied after embedding and between layers

### 🔹 Decoder
- Input: Encoder’s final hidden state + target (for teacher forcing)
- Embedding → RNN → Linear → Log-Softmax
- Dropout applied similarly

### 🔹 Seq2Seq
- Combines Encoder and Decoder
- Training uses **teacher forcing**
- Inference uses **beam search** (or greedy decoding)

---

## ⚙️ Setup

###  Prerequisites

- Environment: Kaggle notebook with GPU
- Install:
  ```bash
  pip install torch pandas wandb tqdm
  ```

- Wandb:
  ```bash
  wandb login
  ```

- Dataset should be available in the given Kaggle directory.

---

##  Hyperparameter Sweep

###  Method: Bayesian Optimization (`method: 'bayes'`)

**Sweep Search Space (defined in `train_loader.py`):**
```python
{
  emb_dim: [64, 128, 256],
  hidden_dim: [128, 256],
  enc_layers: [1, 2, 3],
  dec_layers: [1, 2, 3],
  cell_type: ['LSTM', 'GRU', 'RNN'],
  dropout: [0.2, 0.3, 0.4],
  batch_size: [32, 64, 128],
  learning_rate: [0.001, 0.0005, 0.0001],
  teacher_forcing: [0.5, 0.7, 0.9],
  beam_size: [1, 3, 5],
  patience: 3,
  epochs: [10, 15]
}
```
### Training
  For training, a sweep was generated using the above configuration. A dedicated `Train` class is provided, which includes all functions for training and evaluation. Additionally, a `train_loader` function is implemented to handle the sweep execution.



##  Best Hyperparameters


```python
config = {
    'emb_dim': 128,
    'hidden_dim': 256,
    'enc_layers': 2,
    'dec_layers': 3,
    'cell_type': 'GRU',
    'dropout': 0.4,
    'batch_size': 64,
    'learning_rate': 0.0005,
    'teacher_forcing': 0.7,
    'beam_size': 1,
    'epochs': 10,
    'patience': 3
}
```

---

##  Test evaluator
A `TestEvaluator` class is also defined, containing functions such as `compute_sequence_prediction` and `generate_sample` for evaluating the model. Following this, a `test_loader` function is implemented to handle evaluation with the best hyperparameters from the sweep — it retrains the model using these parameters, evaluates it on the test set, and returns both the prediction file and the final trained model.


---

---

##  Results

| Metric              | Value (approx.) |
|---------------------|-----------------|
| Val Sequence Accuracy |  30-35%       |
| Test Sequence Accuracy | 35%       |


---

## 📌 How to Use

1. **Create Kaggle notebook**
2. **Install dependencies and login to Wandb**
3. ** Finding best hyperparamter run upto first config after that find best hyperparamter from wandb and train train again with these hyperparameter and test and get the output
   ``` Run the coressponding cell 

## 📂 Output

- `best_model.pt`: Trained model checkpoint
- `predictions.tsv`: Full test predictions
