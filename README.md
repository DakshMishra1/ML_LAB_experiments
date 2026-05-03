# 🤖 AI & ML Lab Experiments

A comprehensive collection of **Applied Machine Learning** and **Artificial Intelligence** experiments covering classification, regression, NLP, and deep learning tasks. This repository contains hands-on implementations of various ML algorithms and techniques for real-world problem-solving.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## 🎯 Overview

This lab repository contains practical implementations of machine learning models for diverse applications including:

- **Classification Tasks**: Binary and multi-class classification problems
- **Regression Tasks**: Continuous value prediction
- **Natural Language Processing**: Text analysis and sentiment detection
- **Deep Learning**: Neural networks for image recognition
- **Time Series Analysis**: Stock price and trend prediction
- **Anomaly Detection**: Identifying unusual patterns in data

Each experiment includes:
✅ Data preprocessing and exploration  
✅ Model training and evaluation  
✅ Performance metrics and visualizations  
✅ Output reports and charts  

---

## 📁 Projects

### 1. **Breast Cancer Diagnosis** 🏥
**File**: `BreastCancerDiagnosis.py`
- **Objective**: Predict whether a tumor is malignant or benign
- **Dataset**: sklearn Breast Cancer Dataset
- **Models**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN
- **Output**: Confusion matrices, ROC curves, classification reports

### 2. **Iris Flower Classification** 🌸
**File**: `Flower_Classification.py`
- **Objective**: Classify iris flowers into three species
- **Dataset**: Iris dataset
- **Model**: Logistic Regression
- **Output**: Accuracy scores and classification metrics

### 3. **Handwritten Digit Recognition** 🔢
**File**: `HandwrittenDigitRecognition.py`
- **Objective**: Recognize handwritten digits (0-9) from images
- **Dataset**: MNIST dataset
- **Architecture**: Deep Neural Network with Keras
- **Model**: Flatten → Dense(128) → Dense(10) with softmax activation
- **Output**: Model accuracy and training history

### 4. **Sentiment Analysis** 😊😞
**File**: `SentimentAnalysis.py`
- **Objective**: Classify text reviews as positive or negative
- **Techniques**: TF-IDF vectorization, NLP preprocessing (stopwords removal)
- **Models**: Logistic Regression, Naive Bayes, LinearSVC
- **Output**: Confusion matrices, accuracy comparisons

### 5. **Spam Detection** 📧
**File**: `spam_detection.py`
- **Objective**: Identify spam emails or messages
- **Preprocessing**: Text cleaning, stopword removal
- **Models**: Logistic Regression, Naive Bayes, LinearSVC
- **Output**: Spam detection performance metrics

### 6. **Customer Churn Prediction** 📊
**File**: `customer_churn_prediction.py`
- **Objective**: Predict which customers are likely to leave
- **Models**: Logistic Regression, Decision Tree, Random Forest
- **Evaluation**: ROC curves, confusion matrices, classification reports
- **Output**: Churn prediction analysis and visualizations

### 7. **Fake News Detection** 📰
**File**: `fake_news_detection.py`
- **Objective**: Classify news articles as real or fake
- **Techniques**: TF-IDF vectorization, text feature extraction
- **Models**: Logistic Regression, Naive Bayes, LinearSVC
- **Output**: Model performance metrics and comparisons

### 8. **House Price Prediction** 🏠
**File**: `house_prediction.py`
- **Objective**: Predict house prices based on features
- **Dataset**: Housing dataset with property attributes
- **Model**: Linear Regression
- **Metrics**: MSE, R² score

### 9. **Stock Price Prediction** 📈
**File**: `stockPricePrediction.py`
- **Objective**: Predict stock price movements
- **Data Source**: Yahoo Finance (via yfinance library)
- **Models**: Linear Regression, Random Forest Regressor
- **Preprocessing**: MinMax scaling, time series formatting
- **Output**: Price predictions, trend visualizations

### 10. **Traffic Sign Recognition** 🚦
**File**: `traffic_sign_recognition.py`
- **Objective**: Classify traffic signs from images
- **Dataset**: Digits dataset (8x8 images)
- **Models**: Random Forest, Logistic Regression, SVM
- **Output**: Classification accuracy and confusion matrices

### 11. **Credit Risk Classification** 💳
**File**: `credit_risk.py`
- **Objective**: Assess credit risk for loan applicants
- **Models**: Classification algorithms for risk evaluation
- **Output**: Risk prediction and confidence scores

### 12. **Disease Diagnosis** 🔬
**File**: `disease_diagnosis.py`
- **Objective**: Predict disease occurrence based on medical indicators
- **Approach**: Supervised learning for diagnostic classification
- **Output**: Diagnosis predictions and confidence metrics

### 13. **Anomaly Detection** 🎯
**File**: `anomaly_detection`
- **Objective**: Identify unusual patterns in datasets
- **Techniques**: Statistical and ML-based anomaly detection
- **Output**: Anomaly scores and detection reports

### 14. **Movie Recommendation System** 🎬
**File**: `movie_recommend.py`
- **Objective**: Recommend movies based on user preferences
- **Approach**: Collaborative filtering or content-based recommendation
- **Output**: Personalized movie suggestions

---

## 📦 Prerequisites

- **Python 3.7+**
- **Jupyter Notebook** (optional, for interactive exploration)
- **Libraries**:
  - `scikit-learn` - Machine learning algorithms
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `matplotlib` & `seaborn` - Data visualization
  - `tensorflow` & `keras` - Deep learning
  - `nltk` - Natural language processing
  - `yfinance` - Financial data download

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/AI-ML-Lab.git
cd AI-ML-Lab/aimlLab
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn tensorflow nltk yfinance
```

---

## 🚀 Usage

### Run Any Experiment
```bash
python <experiment_filename>.py
```

**Example:**
```bash
# Run Breast Cancer Diagnosis
python BreastCancerDiagnosis.py

# Run Sentiment Analysis
python SentimentAnalysis.py

# Run Stock Price Prediction
python stockPricePrediction.py
```

### Output
Each experiment generates:
- **Console Output**: Model performance metrics, accuracy scores
- **Visualization Plots**: Saved in `outputs/` directory
- **Classification Reports**: Detailed precision, recall, F1-scores
- **Confusion Matrices**: Visual representation of prediction accuracy

---

## 📂 Project Structure

```
aimlLab/
├── BreastCancerDiagnosis.py          # Classification (5 models)
├── Flower_Classification.py           # Iris classification
├── HandwrittenDigitRecognition.py    # Deep Learning (MNIST)
├── SentimentAnalysis.py              # NLP - Sentiment classification
├── spam_detection.py                 # Text classification
├── customer_churn_prediction.py      # Customer analytics
├── fake_news_detection.py            # News classification
├── house_prediction.py               # Regression
├── stockPricePrediction.py           # Time series prediction
├── traffic_sign_recognition.py       # Image classification
├── credit_risk.py                    # Risk classification
├── disease_diagnosis.py              # Medical diagnosis
├── anomaly_detection                 # Anomaly detection
├── movie_recommend.py                # Recommendation system
├── outputs/                          # Generated visualizations
└── README.md                         # This file
```

---

## 🔧 Key Technologies & Libraries

| Technology | Purpose |
|-----------|---------|
| **scikit-learn** | ML algorithms (classification, regression) |
| **TensorFlow/Keras** | Deep learning models |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical operations |
| **Matplotlib/Seaborn** | Data visualization |
| **NLTK** | Natural language processing |
| **yfinance** | Financial data fetching |

---

## 📊 Common Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: MSE, RMSE, MAE, R² Score
- **Visualization**: Confusion Matrix, ROC Curve, Feature Importance

---

## 💡 Learning Outcomes

Through these experiments, you will learn:

✅ Data preprocessing and feature engineering  
✅ Model selection and hyperparameter tuning  
✅ Classification and regression techniques  
✅ Natural Language Processing fundamentals  
✅ Deep learning with neural networks  
✅ Time series analysis and forecasting  
✅ Performance evaluation and metrics interpretation  
✅ Data visualization best practices  

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/new-experiment`)
3. Add your experiment with documentation
4. Commit your changes (`git commit -m 'Add new ML experiment'`)
5. Push to the branch (`git push origin feature/new-experiment`)
6. Open a Pull Request

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Author

**Daksh Mishra**  
B.Tech CSE - Applied Machine Learning Laboratory

---

## 📚 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

---

## ⚡ Quick Start Guide

```bash
# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn tensorflow nltk yfinance

# Run an experiment
python BreastCancerDiagnosis.py

# Check outputs
# Results will be saved in outputs/ directory
```

---

## 📞 Support

For issues, questions, or suggestions, feel free to:
- Open an issue on GitHub
- Create a discussion thread
- Contact the author

---

**Happy Learning! 🚀**
