# 📈 AI-Powered Stock Price Prediction with LSTM

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [License](#-license)
- [Contact](#-contact)

## 🌟 Project Overview
An end-to-end machine learning pipeline for stock price forecasting using:
- Yahoo Finance API for real-time data
- LSTM neural networks for time-series prediction
- Streamlit for interactive web deployment

## ✨ Key Features
✅ Real-time stock data fetching  
✅ Technical indicators (RSI, Moving Averages)  
✅ LSTM model with TensorFlow/Keras  
✅ Interactive web dashboard  
✅ Automated CI/CD pipeline  

## 🛠️ Tech Stack
| Component          | Technology |
|--------------------|------------|
| Data Processing    | Pandas, NumPy |
| Machine Learning   | TensorFlow, scikit-learn |
| Visualization      | Matplotlib, Seaborn |
| Deployment         | Streamlit |

## 💻 Installation

1. Clone the repository:
```bash
  git clone https://github.com/yourusername/stock-price-prediction.git
  cd stock-price-prediction
```


2. Install dependencies 
```bash
  pip install -r requirements.txt
```

## 🚀 Usage
To train the model:
python src/train.py

To launch the web app:
streamlit run src/app.py

## 📂 Project Structure
stock-price-prediction/
├── data/               # Sample datasets
├── notebooks/          # Jupyter notebooks for EDA
├── src/
│   ├── train.py        # Model training script
│   ├── app.py          # Streamlit application
│   └── utils.py        # Helper functions
├── models/             # Saved model weights
├── requirements.txt    # Dependency list
└── README.md           # Project documentation

📧 Contact
Janhavi Gurav - 4janhavig@gmail.com

Project Link: https://github.com/yaarrjanhavi/stock-price-prediction
