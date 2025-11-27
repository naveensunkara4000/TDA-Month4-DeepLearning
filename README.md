
# 📘 TDA Month 4 (Advanced Deep Learning & NLP)



---

## 📖 Overview  
This repository contains Week 12–13 projects from the TDA Month 4 module, focusing on Advanced Deep Learning and sequence modeling techniques.
The tasks include implementing CNNs for image classification and LSTM networks for time series forecasting, using TensorFlow, Keras, NumPy, Pandas, and Matplotlib.

These projects strengthen practical skills in deep learning model building, training, evaluation, visualization, and saving model artifacts.
---

## 🎯 Objectives  
- Understand and implement Convolutional Neural Networks (CNNs).  
- Build and train Recurrent Neural Networks (RNNs) and LSTMs for sequential data.
- Conduct model evaluation using graphs, metrics, and prediction visualization.  
- Perform hands-on learning through real-world style client projects. 
- Document code, outputs, and learning outcomes for submission.

---

## 🗂️ Project Structure
```bash

tda_month4/
├── data/
│   └── stock_prices.csv        # (optional) real time series data
├── models/
│   ├── cnn_cifar10.h5       
│   ├── lstm_timeseries.h5      
│   └── lstm_scaler.pkl         
├── outputs/
│   ├── week12/                
│   └── week13/                
├── week12/
│   └── cnn_cifar10.py         
├── week13/
│   └── lstm_timeseries.py      
├── venv/
├── .gitignore
├── requirements.txt
└── README.md

```

## Installation & Setup
1️⃣ Prerequisites
Ensure you have these installed:
* Python 3.8+
* VS Code
* Git

2️⃣ Clone the Repository
```bash
 git clone https://github.com/naveensunkara4000/TDA-Month4-DeepLearning.git
cd TDA-Month4-DeepLearning

```
3️⃣ Create & Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
```
4️⃣ Install Dependencies
```bash
 pip install -r requirements.txt
```
##  Week-wise Breakdown

### 📦 Week 12 – Convolutional Neural Networks (CNNs)

**Concepts:** 
- Introduction to CNN architecture
- Convolution → ReLU → Pooling → Flatten → Dense
- Feature extraction and image classification
- Using TensorFlow/Keras for CNN model building 

**Hands-on:** 
- loaded CIFAR-10 dataset
- Preprocessed image data
- Built a CNN model with Conv2D, MaxPooling2D, Dense layers
- Trained the model for 10 epochs
- Generated evaluation accuracy/loss graphs
- Visualized predictions for sample test images

**Deliverable:** 
- `cnn_cifar10.py`
- Trained model: `models/cnn_cifar10.h5`
- Output files:
    - `outputs/week12/accuracy.png`
    - `outputs/week12/loss.png`
    - `outputs/week12/sample_predictions.png`
    - `outputs/week12/evaluation.txt`
---

###  📈 Week 13 – RNNs & LSTMs for Time Series Forecasting

**Concepts:** 
 - Introduction to **Recurrent Neural Networks (RNNs)**
 - Vanishing gradient problem and motivation for **LSTMs**
 - Time series modeling and forecasting
 - Sliding window method
 - RMSE evaluation

**Hands-on:**
 - sed real or synthetic time-series data
 - Applied MinMaxScaler
 - Created windowed sequences of 20 timestamps
 - Built an LSTM model using Keras LSTM layer
 - Trained the model on CPU
 - Compared Actual vs Predicted values
 - Generated prediction plots  
 
**Deliverable:** 
  - `lstm_timeseries.py`
  - Trained model:` models/lstm_timeseries.h5`
  - Outputs:
      - `outputs/week13/predictions.png`
      - `outputs/week13/loss.png`
      - `outputs/week13/evaluation.txt` 

---
###  📦 Requirements
`````bash
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib

`````
---
