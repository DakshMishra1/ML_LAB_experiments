# 🤖 ML Lab Experiments

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-green?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p align="center">
  A curated collection of Machine Learning experiments, algorithms, and implementations developed as part of the ML Laboratory coursework.
</p>

---

## 👨‍💻 Author

**Daksh Mishra**
- GitHub: [@DakshMishra1](https://github.com/DakshMishra1)
- Repository: [ML_LAB_experiments](https://github.com/DakshMishra1/ML_LAB_experiments)

---

## 📌 About This Repository

This repository contains hands-on implementations of core Machine Learning algorithms and techniques. Each experiment is organized as a self-contained Jupyter Notebook, making it easy to understand the theory, code, and results together in one place.

---

## 🗂️ Repository Structure

```
ML_LAB_experiments/
│
├── 01_Linear_Regression/
│   └── linear_regression.ipynb
│
├── 02_Logistic_Regression/
│   └── logistic_regression.ipynb
│
├── 03_KNN_Classifier/
│   └── knn_classifier.ipynb
│
├── 04_Decision_Tree/
│   └── decision_tree.ipynb
│
├── 05_Random_Forest/
│   └── random_forest.ipynb
│
├── 06_SVM/
│   └── svm_classifier.ipynb
│
├── 07_Naive_Bayes/
│   └── naive_bayes.ipynb
│
├── 08_K_Means_Clustering/
│   └── kmeans_clustering.ipynb
│
├── 09_PCA/
│   └── pca_dimensionality_reduction.ipynb
│
├── 10_Neural_Network/
│   └── neural_network_basics.ipynb
│
├── datasets/
│   └── (datasets used across experiments)
│
└── README.md
```

---

## 🧪 Experiments Overview

| # | Experiment | Algorithm | Type |
|---|-----------|-----------|------|
| 01 | Linear Regression | Gradient Descent / OLS | Supervised |
| 02 | Logistic Regression | Sigmoid + MLE | Supervised |
| 03 | K-Nearest Neighbours | Distance Metrics | Supervised |
| 04 | Decision Tree | CART / ID3 | Supervised |
| 05 | Random Forest | Ensemble Bagging | Supervised |
| 06 | Support Vector Machine | Kernel Trick | Supervised |
| 07 | Naive Bayes | Bayes Theorem | Supervised |
| 08 | K-Means Clustering | Centroid-based | Unsupervised |
| 09 | PCA | Eigen Decomposition | Dimensionality Reduction |
| 10 | Neural Network | Backpropagation | Deep Learning |

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Notebook**: Jupyter Notebook / Google Colab
- **Libraries**:
  - `numpy` — Numerical computing
  - `pandas` — Data manipulation
  - `matplotlib` & `seaborn` — Visualization
  - `scikit-learn` — ML algorithms & utilities
  - `tensorflow` / `keras` *(for neural networks)*

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/DakshMishra1/ML_LAB_experiments.git
cd ML_LAB_experiments
```

### 2. Create a Virtual Environment *(recommended)*

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then navigate to the desired experiment folder and open the `.ipynb` file.

---

## ☁️ Run on Google Colab

You can also run any notebook directly on Google Colab without any local setup:

1. Open [Google Colab](https://colab.research.google.com/)
2. Click **File → Open notebook → GitHub**
3. Paste the repository URL: `https://github.com/DakshMishra1/ML_LAB_experiments`
4. Select the desired notebook and run!

---

## 📊 Sample Results

Each notebook includes:
- 📖 **Theory** — Brief explanation of the algorithm
- 🧹 **Data Preprocessing** — Cleaning and preparing the dataset
- 🏋️ **Model Training** — Fitting the model to training data
- 📈 **Evaluation** — Accuracy, confusion matrix, classification report
- 📉 **Visualization** — Plots of decision boundaries, clusters, loss curves, etc.

---

## 📋 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
tensorflow>=2.6.0
```

> You can generate `requirements.txt` by running: `pip freeze > requirements.txt`

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve an experiment or add a new one:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/new-experiment`
3. Commit your changes: `git commit -m "Add: new experiment"`
4. Push to the branch: `git push origin feature/new-experiment`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with proper attribution.

---

## ⭐ Show Your Support

If you find this repository helpful, please consider giving it a **star ⭐** on GitHub — it keeps the motivation going!

---

<p align="center">Made with ❤️ by <a href="https://github.com/DakshMishra1">Daksh Mishra</a></p>
