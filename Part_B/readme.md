
# Bengali Transliteration with Attention Mechanism

This project implements a character-level Seq2Seq model for transliterating words from Latin script (English) to Bengali script using the Dakshina dataset. The model is trained **without an attention mechanism** and optimized using a hyperparameter sweep with **Weights & Biases (Wandb)**.

---

## 🔧 Overview

- **Dataset**: Dakshina (Bengali)
- **Task**: Transliteration (Latin → Bengali)
- **Model**: Encoder–Decoder (RNN/GRU/LSTM)
- **Framework**: PyTorch
- **Evaluation**: Sequence accuracy on test set
- **Hardware**: Kaggle with GPU (if available)

---

## 📁 Project Structure

```
without_attention/


---

## 📦 Dataset

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

## 🧠 Model Architecture
### Attention Class

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

### ✅ Prerequisites

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

## 🧪 Hyperparameter Sweep

### 🔍 Method: Bayesian Optimization (`method: 'bayes'`)

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

### 📈 Observations

- **LSTM** consistently outperformed other RNN cells
- **RNN** often failed (<5% accuracy)
- Best batch sizes: **32 or 64**
- Optimal dropout: **0.2–0.4**
- **Beam size = 1** worked best (greedy decoding)
- Teacher Forcing: **0.5–0.9** was effective

---

## 🏆 Best Hyperparameters

From `train_evaluate.py`:

```python
config = {
    'emb_dim': 64,
    'hidden_dim': 256,
    'enc_layers': 2,
    'dec_layers': 3,
    'cell_type': 'LSTM',
    'dropout': 0.4,
    'batch_size': 64,
    'learning_rate': 0.0005,
    'teacher_forcing': 0.9,
    'beam_size': 3,
    'epochs': 15,
    'patience': 3
}
```

---

## 🏋️ Training

- **Script**: `train_evaluate.py`
- **Trainer class** handles model training and validation
- **Early stopping** with patience = 3
- Model saved to: `/kaggle/working/best_model.pt`
- Metrics logged to **Wandb**:
  - `train_loss`, `val_loss`
  - `token_accuracy`, `sequence_accuracy`
-The model is trained using the Adam optimizer with a cross-entropy loss function. During training, the model learns to minimize the difference between the predicted translations and the ground truth translations in the training set.


---

## 🧾 Evaluation

After training, the model is evaluated on the test set to assess its performance on unseen data. The test accuracy and loss are reported to measure the effectiveness of the model.

---

## 📌 How to Use

1. **Create Kaggle notebook**
2. **Install dependencies and login to Wandb**
3. **Run sweep**:
   ```bash
   python train_loader.py
   ```
4. **Train & evaluate best config**:
   ```bash
   python train_evaluate.py
   ```
5. **Evaluate on test set**:
   ```bash
   python test_evaluator.py
   ```

---


---

## 📂 Output

- `best_model.pt`: Trained model checkpoint
- `predictions.tsv`: Full test predictions
- `heatmap.png`: First 9 sample of heat map
- 35 highlighted samples for qualitative evaluation
