🛡️ Intelligent Intrusion Detection System (IDS)
📌 Project Overview

This project aims to design and implement an Intelligent Intrusion Detection System (IDS) based on Machine Learning and Deep Learning techniques to detect abnormal and malicious behaviors in network traffic.

Traditional security mechanisms (firewalls, ACLs, antivirus) are insufficient to detect advanced, subtle, or zero-day attacks. This work proposes a data-driven IDS capable of learning from real network traffic to enhance enterprise cybersecurity and SOC monitoring capabilities.

🎯 Objectives
Main Objective

Develop an intelligent IDS capable of automatically detecting intrusions and anomalous behaviors in enterprise network traffic.

Specific Objectives

Collect and preprocess a representative network traffic dataset

Explore and analyze traffic features and class imbalance

Train and compare multiple ML and DL models

Evaluate models using standard performance metrics

Detect multiple attack categories

Provide visualization and alert analysis via a dashboard

Propose a robust and extensible IDS prototype

🗂️ Dataset

Source: CICIDS2017

Type: Tabular network flow dataset

Features: Extracted using CICFlowMeter

Classes:

BENIGN

Bot

BruteForce

DoS / DDoS

Injection

Scan

⚠️ The dataset is highly imbalanced, with BENIGN traffic largely dominating attack classes.

🔄 Data Processing Pipeline

Data Cleaning

Removal of duplicates

Handling missing and abnormal values

Feature Engineering

Normalization of numerical features

Encoding categorical attributes

Labeling

Binary classification: BENIGN vs ATTACK

Multi-class classification: attack categories

Dataset Split

Training

Validation

Test

🧠 Models Implemented
🔹 Supervised Learning

Random Forest (standard & balanced)

XGBoost

Linear SVM

Logistic Regression

🔹 Unsupervised Learning

Isolation Forest

K-Means

🔹 Deep Learning

TabNet (tabular deep learning)

Multi-Layer Perceptron (MLP)

📊 Model Evaluation
Metrics Used

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC-AUC (binary classification)

Key Results (Summary)
Model	Accuracy (%)	F1-score (%)
Random Forest (balanced)	99.84	99.84
XGBoost	99.78	99.77
TabNet (Binary)	93.0	93.0
TabNet (Multi-class)	93.0	92.0
Isolation Forest	80.87	50.0
KMeans	83.0	46.0

✔️ Best multi-class models: Random Forest, XGBoost
✔️ Best binary detection model: TabNet

📈 Visualization & Interface

A Streamlit dashboard was developed to:

Upload traffic CSV files

Visualize predictions

Compare model performances

Display alerts and anomaly scores

🛠️ Technologies Used

Programming Language: Python 3.x

Libraries:

scikit-learn

XGBoost

PyTorch / TensorFlow

Pandas, NumPy

Matplotlib, Seaborn

Streamlit

Tools:

CICFlowMeter

Jupyter Notebook

Git / GitHub

📁 Project Structure (example)
IDS/
│── data/
│   ├── raw/
│   ├── processed/
│
│── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   ├── modeling.ipynb
│
│── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── tabnet.pth
│
│── app/
│   ├── streamlit_app.py
│
│── README.md
│── requirements.txt

🚀 How to Run the Project

Clone the repository:

git clone https://github.com/mayssaboumaiza/Cyber-Security-Project.git
cd Cyber-Security-Project


Install dependencies:

pip install -r requirements.txt


Launch the Streamlit application:

streamlit run app/streamlit_app.py

📌 Conclusion

This project demonstrates that Machine Learning and Deep Learning techniques significantly improve intrusion detection capabilities compared to traditional security mechanisms.

Supervised models provide high precision for known attacks

Deep learning models offer better generalization for unseen threats

Combining ML and DL leads to a robust and adaptive IDS

🔮 Future Work

Improve detection of rare attack classes (SMOTE, ADASYN)

Integrate time-series and protocol-specific features

Ensemble learning (RF + TabNet)

Real-time streaming detection

Integration with SIEM / SOC platforms

👩‍💻 Authors

Mayssa Boumaiza

Hedil Jlassi

Supervised by: Mr. Hermaci Houcem
Academic Year: 2025–2026
