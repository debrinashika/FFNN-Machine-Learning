# Feedforward Neural Network (FFNN)  

## Description  
A **Feedforward Neural Network (FFNN)** is a type of artificial neural network where information flows in one direction, from input to output, without loops or feedback connections.  

This project implements an FFNN from scratch, including the **feedforward and backpropagation algorithms**, with support for various activation functions such as **Linear, ReLU, Sigmoid, Tanh, Softmax, ELU, and Swish**. Additionally, it supports multiple loss functions, including **Mean Squared Error (MSE), Binary Cross-Entropy, and Categorical Cross-Entropy**.  

The model is customizable with parameters such as **batch size, learning rate, number of epochs, and verbosity mode**.  

Furthermore, the model supports **saving and loading trained weights**, allowing for reuse without retraining and **graph visualization**.

## Setup Instructions  

### 1. Clone the Repository  
```bash
git clone https://github.com/debrinashika/FFNN-Machine-Learning.git
cd FFNN-Machine-Learning
```
### 2. Configure Model Parameters  
If you want to modify parameters such as **batch size, learning rate, number of epochs, or verbosity mode**, edit the **`main.py`** file in the following section:  

```python
ffnn = FFNN(batch_size=5000, learning_rate=0.01, epoch=3, verbose=1, loss_func='mse', weight_init='normal', seed=42)
```
- **batch_size**: Number of samples processed in one iteration  
- **learning_rate**: The rate at which the model learns  
- **epoch**: Number of training iterations  
- **verbose**:  
  - `0`: No output during training  
  - `1`: Displays a progress bar with training and validation loss  

### 3. Train the Model  
```bash
python main.py
```
> This will train the model and save it as `ffnn_mnist_model.pkl`.  

### 4. Load a Trained Model  
If you have already trained and saved the model, you can **load it for predictions** without retraining:  

```python
import pickle
import numpy as np

# Load the trained model
with open("ffnn_mnist_model.pkl", "rb") as f:
    loaded_ffnn = pickle.load(f)

# Make predictions
y_pred = loaded_ffnn.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Evaluate accuracy
accuracy = np.mean(y_pred_labels == y_test) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
```

## Project Structure  
```
FFNN-Machine-Learning/
│── src/                 # FFNN implementation
│   ├── ffnn.py
│   ├── layers.py
│   ├── activations.py
│   ├── backpropagation.py
│   ├── main.py
│   ├── Tubes_ML.ipynb
│── README.md             
```

## Task Distribution  

| NIM | Name | Responsibilities |
|-------------|-------------|-----------------|
| 13522009 | Muhammad Yusuf Rafi | Membuat fungsi save & load, Membuat fungsi distribusi bobot dan gradien, Implementasi Fungsi Aktivasi linear, dan implementasi Fungsi Loss Categorical Cross-Entropy |
| 13522025 | Debrina Veisha Rashika | Membuat model FFNN, Inisialisasi bobot, Implementasi Fungsi Aktivasi Tanh dan Softmax, dan Implementasi Fungsi Loss MSE |
| 13522035 | Melati Anggraini | Membuat visualisasi graf, Implementasi Fungsi Aktivasi ReLU dan Sigmoid, dan Implementasi Fungsi Loss Binary Cross-Entropy |
---
