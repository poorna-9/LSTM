# 🔁 NeuroLite-LSTM

A minimal, transparent, and fully-customizable Long Short-Term Memory (LSTM) network framework built entirely from scratch using **NumPy**. Designed for deep learning enthusiasts and learners who want to understand how LSTMs work at the cell and gate level.

## 🚀 Features

- Fully working **LSTM cell** implementation from scratch (no frameworks).
- Gate-level operations: **Forget Gate, Input Gate, Output Gate, Cell State updates**.
- Forward and backward pass (Backpropagation Through Time - BPTT) implemented manually.
- Modular and extendable design using Python classes.
- Supports:
  - **Many-to-one** and **many-to-many** LSTM outputs.
  - Multiple LSTM layers and custom layer stacking.
  - Custom **activation** and **loss** functions.
- Trained on **sequence prediction tasks** and small text datasets.
- Built using only **NumPy**.

## 📊 Applications

- Sequence prediction (e.g., next character prediction)
- Text classification
- Time-series forecasting
- Any other task requiring memory of previous inputs

## 🧩 Example Usage

```python
from neuro_lite_lstm import LSTM, Dense, Tanh, Softmax, CrossEntropyLoss, NeuralNetwork

model = NeuralNetwork()
model.add(LSTM(input_size=10, hidden_size=64))  # e.g. input seq of length 10
model.add(Dense(64, 32))
model.add(Tanh())
model.add(Dense(32, vocab_size))
model.add(Softmax())

model.compile(loss=CrossEntropyLoss(), learning_rate=0.01)
model.fit(X_train, y_train, epochs=20, batch_size=1)
model.evaluate(X_test, y_test)
