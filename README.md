# SMS Spam Classification Using Machine Learning

## Project Overview

This project implements an **SMS Spam Classification System** using Natural Language Processing (NLP) and five different Machine Learning algorithms.

The objective is to classify SMS messages into two categories:

- **Ham** – legitimate/non-spam message
- **Spam** – unwanted or fraudulent message

The project uses the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

Five machine learning algorithms are implemented and compared:

1. AdaBoost Classifier
2. Gradient Boosting Classifier
3. XGBoost Classifier
4. Extra Trees Classifier
5. Ridge Classifier

The models are evaluated using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## Dataset

### SMS Spam Collection Dataset

The dataset contains **5,574 SMS messages** labeled as either `ham` or `spam`.

| Class | Description |
|---|---|
| Ham | Legitimate SMS |
| Spam | Unwanted/spam SMS |

### Dataset Source

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/228/sms

### Dataset Citation

Almeida, T. A., & Hidalgo, J. M. (2011).  
**SMS Spam Collection [Dataset]**.  
UCI Machine Learning Repository.  
DOI: 10.24432/C5CC84.

---

## Project Objectives

The main objectives of this project are:

- To understand SMS spam classification using NLP.
- To preprocess and clean text data.
- To convert text into numerical features using TF-IDF.
- To train multiple machine learning classification algorithms.
- To evaluate the performance of different algorithms.
- To compare the models using standard evaluation metrics.
- To identify the best-performing algorithm for SMS spam detection.

---

## Algorithms Used

### 1. AdaBoost

AdaBoost, or Adaptive Boosting, combines multiple weak learners to create a stronger classifier.

It gives greater importance to incorrectly classified training samples during subsequent iterations.

```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)
```

---

### 2. Gradient Boosting

Gradient Boosting builds models sequentially. Each new model attempts to reduce the errors made by the previous models.

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)
```

---

### 3. XGBoost

XGBoost is an optimized gradient boosting algorithm that is widely used for classification and regression problems.

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)
```

---

### 4. Extra Trees Classifier

Extra Trees, or Extremely Randomized Trees, is an ensemble learning algorithm that constructs multiple randomized decision trees.

```python
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42
)
```

---

### 5. Ridge Classifier

Ridge Classifier is a linear classification algorithm based on Ridge regression with L2 regularization.

It is particularly suitable for high-dimensional sparse text features such as TF-IDF.

```python
from sklearn.linear_model import RidgeClassifier

model = RidgeClassifier()
```

---

## Technologies Used

The project is implemented using Python.

### Programming Language

- Python 3.x

### Libraries

- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Project Structure

```text
SMS-Spam-Classification/
│
├── README.md
│
├── SMS_Spam_Classification.ipynb
│
├── SMSSpamCollection.csv
│
├── SMS_Spam_NLP_5_Algorithms_Lab_Report.docx
│
└── results/
    ├── confusion_matrices/
    └── comparison_graphs/
```

---

## Installation

Make sure Python is installed on your computer.

Install the required libraries using:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn jupyter
```

Alternatively:

```bash
pip install -r requirements.txt
```

---

## Requirements

The project requires the following packages:

```text
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
jupyter
```

---

## Running the Project

### Step 1: Clone or Download the Project

Download the project files to your computer.

### Step 2: Install Dependencies

Run:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn jupyter
```

### Step 3: Place the Dataset

Place the SMS dataset in the project directory.

The expected dataset file is:

```text
SMSSpamCollection.csv
```

### Step 4: Start Jupyter Notebook

Run:

```bash
jupyter notebook
```

### Step 5: Open the Notebook

Open:

```text
SMS_Spam_Classification.ipynb
```

### Step 6: Run All Cells

In Jupyter Notebook, select:

```text
Kernel → Restart & Run All
```

The notebook will preprocess the dataset, train all five models, calculate evaluation metrics, and generate comparison graphs and confusion matrices.

---

## Data Preprocessing

The following preprocessing operations are performed:

1. Load the dataset.
2. Remove unnecessary columns.
3. Rename columns.
4. Check for missing values.
5. Convert labels into numerical form.
6. Convert text to lowercase.
7. Remove unnecessary newline characters.
8. Split the dataset into training and testing sets.

Example:

```python
df["message"] = df["message"].str.lower()
```

Labels are encoded as:

```text
ham  → 0
spam → 1
```

The dataset is divided using an 80:20 train-test split.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## NLP and TF-IDF

Since machine learning algorithms cannot directly process raw text, SMS messages are converted into numerical feature vectors.

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

TF-IDF assigns higher importance to words that are useful for distinguishing between ham and spam messages.

---

## Model Training

All five algorithms are trained using the processed TF-IDF features.

The general workflow is:

```text
SMS Dataset
     ↓
Data Cleaning
     ↓
Train-Test Split
     ↓
TF-IDF Feature Extraction
     ↓
Model Training
     ↓
Prediction
     ↓
Evaluation
     ↓
Model Comparison
```

---

## Evaluation Metrics

The models are evaluated using the following metrics.

### Accuracy

Measures the percentage of correctly classified messages.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many messages predicted as spam are actually spam.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual spam messages are correctly identified.

```text
Recall = TP / (TP + FN)
```

### F1-Score

F1-score is the harmonic mean of Precision and Recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Confusion Matrix

The confusion matrix contains:

```text
                 Predicted
               Ham    Spam
Actual Ham      TN      FP
Actual Spam     FN      TP
```

---

## Model Comparison

The project compares the performance of all five algorithms.

An example comparison format is:

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| AdaBoost | 0.96 | 0.91 | 0.79 | 0.85 |
| Gradient Boosting | 0.95 | 0.90 | 0.72 | 0.80 |
| XGBoost | 0.96 | 0.92 | 0.78 | 0.84 |
| Extra Trees | 0.97 | 0.98 | 0.76 | 0.85 |
| Ridge Classifier | 0.98 | 0.97 | 0.88 | 0.92 |

**Note:** The values above are representative/illustrative values for the report format. The exact values should be obtained by running the notebook on the dataset.

---

## Expected Result

The trained models should be able to distinguish spam messages from legitimate messages with high accuracy.

Because the SMS dataset is imbalanced, accuracy alone should not be used to determine the best model.

The **F1-score, precision, and recall** are especially important.

A model with high spam recall is useful because it reduces the number of spam messages incorrectly classified as legitimate messages.

Based on the representative results in this project, **Ridge Classifier** is expected to provide strong performance on TF-IDF text features. However, the final best model should be selected from the actual execution results.

---

## Output

The notebook generates:

- Dataset information
- Dataset statistics
- Class distribution
- Text preprocessing results
- TF-IDF feature representation
- Predictions
- Accuracy
- Precision
- Recall
- F1-score
- Classification reports
- Confusion matrices
- Model comparison table
- Performance comparison graphs

---

## Confusion Matrix

Confusion matrices are generated for each algorithm.

Example:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

---

## Performance Graph

A bar chart can be used to compare the algorithms.

```python
results.plot(
    x="Algorithm",
    y=["Accuracy", "Precision", "Recall", "F1-Score"],
    kind="bar",
    figsize=(12, 6)
)

plt.title("Machine Learning Algorithm Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()
```

---

## Applications

SMS spam classification can be used in:

- Mobile messaging applications
- Email and SMS filtering systems
- Telecommunication networks
- Fraud detection systems
- Customer communication platforms
- Automated message filtering
- Cybersecurity systems
- Marketing message filtering

---

## Advantages

- Automatically identifies spam messages.
- Reduces manual message filtering.
- Can process large numbers of messages.
- TF-IDF provides an effective text representation.
- Multiple machine learning algorithms can be compared.
- The system can be extended to other text classification problems.

---

## Limitations

- The dataset contains relatively short SMS messages.
- New spam patterns may not be represented in the training data.
- Text preprocessing can affect model performance.
- The dataset is class-imbalanced.
- Some algorithms require more computational resources.
- TF-IDF does not fully understand the semantic meaning of words.
- Performance may vary with different datasets.

---

## Future Enhancements

The project can be improved by:

- Using word embeddings.
- Applying Word2Vec.
- Applying GloVe.
- Using BERT or other transformer models.
- Performing hyperparameter tuning.
- Applying cross-validation.
- Using advanced text preprocessing.
- Handling class imbalance using suitable techniques.
- Creating a web-based spam detection application.
- Developing a real-time SMS spam detection system.

---

## Conclusion

This project demonstrates the use of Natural Language Processing and Machine Learning for SMS spam classification.

The SMS messages are cleaned and converted into numerical features using TF-IDF. Five machine learning algorithms—AdaBoost, Gradient Boosting, XGBoost, Extra Trees Classifier, and Ridge Classifier—are trained and evaluated.

The performance of the models is compared using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

The project demonstrates that machine learning can effectively identify spam messages and can serve as a foundation for more advanced NLP-based spam detection systems.

---

## References

1. Almeida, T. A., & Hidalgo, J. M. (2011). SMS Spam Collection. UCI Machine Learning Repository.

2. UCI Machine Learning Repository – SMS Spam Collection:
   https://archive.ics.uci.edu/dataset/228/sms

3. Scikit-learn Documentation:
   https://scikit-learn.org/

4. XGBoost Documentation:
   https://xgboost.readthedocs.io/

5. Python Documentation:
   https://docs.python.org/

---

## Author

**Project:** SMS Spam Classification Using Machine Learning

**Algorithms:**

- AdaBoost
- Gradient Boosting
- XGBoost
- Extra Trees Classifier
- Ridge Classifier

**Domain:** Natural Language Processing / Machine Learning

**Dataset:** UCI SMS Spam Collection