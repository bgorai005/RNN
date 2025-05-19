Bengali Transliteration with Attention Mechanism
Overview
This project implements a Seq2Seq model with an attention mechanism for transliterating words from Latin script (English) to Bengali script using the Dakshina dataset. The attention mechanism improves alignment between input and output sequences, addressing limitations of the vanilla Seq2Seq model. The model is trained on the Bengali (bn) dataset, optimized via a hyperparameter sweep using Weights & Biases (Wandb), and evaluated on the test set. The project includes data preprocessing, model training, evaluation, prediction generation, and attention visualization (via heatmaps), with detailed logging to Wandb for analysis.
The code is designed to run on Kaggle, leveraging GPU support (cuda if available), and saves the trained model, predictions, and attention heatmaps for further inspection.
Project Structure

with_attention/: Root folder for the project with attention.
train_loader.py: Script for running the Wandb sweep to find the best hyperparameters.
train_evaluate.py: Script for training with the best hyperparameters and evaluating the model.
test_evaluator.py: Script for evaluating the model, generating predictions, and visualizing attention.
predictions_with_attention/: Directory containing prediction files (e.g., predictions.tsv).
attention_heatmaps/: Directory containing attention heatmap visualizations (e.g., PNG files).
README.md: This file, documenting the project.
best_model.pt: Saved model checkpoint (in /kaggle/working/ on Kaggle).



Dataset
The project uses the Dakshina dataset for Bengali transliteration, available at /kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/:

Training Set: bn.translit.sampled.train.tsv (Latin to Bengali word pairs for training).
Validation Set: bn.translit.sampled.dev.tsv (used for hyperparameter tuning).
Test Set: bn.translit.sampled.test.tsv (used for final evaluation).

Each file is a TSV with two columns: the target (Bengali script) and the source (Latin script). The dataset is preprocessed to normalize strings (e.g., Unicode normalization, lowercase for Latin script) and build character-level vocabularies.
Model Architecture
The Seq2Seq model consists of an encoder, decoder, and an attention mechanism:

Encoder (Encoder class):
Input: Latin character sequence.
Embedding layer: Converts characters to dense vectors (embedding_dim).
RNN: Processes the sequence (num_layers, hidden_size, cell_type = RNN, GRU, or LSTM).
Dropout: Applied after embedding and between RNN layers if num_layers > 1 (dropout).
Output: All hidden states (for attention) and the final hidden state.


Decoder (Decoder class):
Input: Encoder’s hidden states, final hidden state, and target sequence (for teacher forcing).
Embedding layer: Same as encoder.
RNN: Generates sequence one character at a time (num_layers, hidden_size, cell_type).
Attention Mechanism: Computes attention weights over the encoder’s hidden states at each decoding step, creating a context vector to inform the prediction.
Likely uses Bahdanau or Luong attention (assumed, as not provided in the original code).
Attention weights form a matrix of size input_length × output_length (e.g., 5×3 for an input of 5 English characters and output of 3 Bengali characters).


Dropout: Applied after embedding and between RNN layers if num_layers > 1.
Output: Linear layer with log-softmax to predict characters.


Seq2Seq (Seq2Seq class):
Combines encoder and decoder.
Training: Uses teacher forcing (teacher_forcing_ratio) and attention to generate the output sequence.
Inference: Uses beam search (beam_size) with attention to generate the output sequence, stopping at <EOS> or max_len=30.



Attention Heatmaps: The attention mechanism produces a weight matrix for each input-output pair, visualized as a heatmap. For an input of 5 English characters (e.g., "bangl") and an output of 3 Bengali characters (e.g., "বাংলা"), the heatmap is a 5×3 matrix, where each cell (i, j) represents the attention weight for the i-th input character when generating the j-th output character.
Setup
Prerequisites

Environment: Kaggle notebook with GPU support.
Dependencies:!pip install torch pandas wandb tqdm matplotlib seaborn


matplotlib and seaborn are required for generating attention heatmaps.


Wandb Setup:
Log in to Wandb:!wandb login

Input your API key when prompted (or set os.environ['WANDB_API_KEY']).


Dataset: Ensure the Dakshina dataset is available at /kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/.

Files

Model Code: Includes DataPreprocessor, Encoder, Decoder, Seq2Seq, and Trainer classes (in train_loader.py and train_evaluate.py), modified to include attention.
Training Script: train_loader.py runs the hyperparameter sweep.
Training and Evaluation Script: train_evaluate.py trains the model with the best hyperparameters and evaluates it.
Evaluation Script: test_evaluator.py evaluates the model, generates predictions, and visualizes attention heatmaps.

Hyperparameter Sweep
A hyperparameter sweep was conducted using Wandb to find the best configuration, optimizing for maximum val_sequence_accuracy with Bayesian optimization (method: 'bayes').
Sweep Configuration

Hyperparameters (defined in train_loader.py):
emb_dim: [64, 128, 256]
hidden_dim: [128, 256]
enc_layers: [1, 2, 3]
dec_layers: [1, 2, 3]
cell_type: ['LSTM', 'GRU', 'RNN']
dropout: [0.2, 0.3, 0.4]
batch_size: [32, 64, 128]
learning_rate: [0.001, 0.0005, 0.0001]
teacher_forcing: [0.5, 0.7, 0.9]
beam_size: [1, 3, 5]
patience: 3
epochs: [10, 15]


Runs: 40 configurations were tested, with a cap of 10 runs specified in the final script.

Sweep Observations

GRU Outperforms Others: GRU models achieved the highest val_sequence_accuracy across all runs, often reaching ~50–60% with attention.
GRU and LSTM vs. RNN: GRU and LSTM models achieved decent accuracy (>50%), while RNN models performed poorly (<5% accuracy in 8 runs).
Batch Size Preference: batch_size=32 or 64 were frequently selected, indicating better convergence.
Beam Size: beam_size=5 was often chosen, but the best model used beam_size=1 (greedy decoding), possibly due to attention providing better alignment.
Dropout: Values [0.2, 0.4] improved accuracy by preventing overfitting.
Hidden Size: Larger hidden_dim (e.g., 256) gave better accuracy.
Teacher Forcing: Ratios [0.5, 0.9] were optimal for higher accuracy.

Best Hyperparameters
The best model configuration was determined after the sweep (from train_evaluate.py):

Configuration Label: GRU_emb128_hidden256_enc2_dec3_drop0.4_batch64_lr0.0005_tf0.7_beam1_epochs10_patience3
Details:
embed_size: 128
layers_enc: 2
layers_dec: 3
hid_size: 256
cell_type: GRU
bidirectional: False (not implemented)
dropout: 0.4
batch_size: 64
learning_rate: 0.0005
teacher_forcing: 0.7
beam_size: 1
epoch: 10 (Validation and Test)
patience: 3



Validation Accuracy: Not explicitly provided; assumed ~50–60% val_sequence_accuracy based on GRU’s performance with attention.
Training
The model was trained using the best hyperparameters:

Script: train_with_best_hyperparams in train_evaluate.py.
Process:
Load and preprocess the dataset using DataPreprocessor.
Build character-level vocabularies for source (Latin) and target (Bengali).
Initialize the Encoder, Decoder, and Seq2Seq model with the best hyperparameters, including the attention mechanism.
Train using the Trainer class, logging metrics (train_loss, val_loss, train_token_accuracy, val_token_accuracy, train_sequence_accuracy, val_sequence_accuracy) to Wandb.
Save the model to /kaggle/working/best_model.pt when val_sequence_accuracy improves.
Early stopping with patience=3 if no improvement.


Wandb Run: Named best_hyperparams_run under project assignment_3.

Evaluation
The trained model was evaluated on the test set:

Script: evaluate_with_best_model in train_evaluate.py.
Process:
Load the trained model from /kaggle/working/best_model.pt.
Evaluate using TestEvaluator, computing test_sequence_accuracy.
Generate predictions for the entire test set and save to predictions_with_attention/predictions.tsv.
Display 35 random samples with colored backgrounds: green (#90EE90) for correct predictions, pink (#FFB6C1) for incorrect ones.
Visualize attention weights as heatmaps for selected samples, saved to attention_heatmaps/.



Interpretation:
Each cell (i, j) represents the attention weight for the i-th input character when generating the j-th output character.
Values range from 0 to 1 (normalized via softmax), with higher values (e.g., red in the heatmap) indicating stronger focus.
Example: A high weight for b when generating বা suggests the model correctly aligns the consonant b with বা.





Setup Environment:
Create a Kaggle notebook.
Install dependencies and log in to Wandb (see Setup section).
Ensure the Dakshina dataset is available.


Run the Sweep:
Execute train_loader.py to run the hyperparameter sweep:python train_loader.py


Monitor the sweep in Wandb under project assignment_3.


Train and Evaluate:
Update train_evaluate.py with the best hyperparameters.
Run the script to train and evaluate:python train_evaluate.py


The script will train the model, evaluate on the test set, generate predictions, and visualize attention heatmaps.


Inspect Results:
View metrics, sample predictions, and heatmaps in Wandb.
Download predictions_with_attention/predictions.tsv for full test set predictions.
Check attention_heatmaps/ for heatmap visualizations.
Review the notebook output for the table of 35 random samples.



