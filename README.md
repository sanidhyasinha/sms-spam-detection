# SMS Spam Detection

This repository contains the code for building a machine learning model to detect **SMS spam messages**. The project uses various classification algorithms and natural language processing (NLP) techniques to classify messages as **spam** or **ham** (non-spam).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Loading the Dataset](#loading-the-dataset)
  - [Data Preprocessing](#data-preprocessing)
  - [Text Vectorization](#text-vectorization)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Model Building](#model-building)
  - [Logistic Regression](#logistic-regression)
  - [Naive Bayes](#naive-bayes)
  - [Random Forest](#random-forest)
- [Results and Evaluation](#results-and-evaluation)
- [Challenges and Future Improvements](#challenges-and-future-improvements)
- [License](#license)

---

## Project Overview

The **SMS Spam Detection** project aims to build a machine learning model that can automatically classify SMS messages as either **spam** or **ham**. By using various machine learning models like Logistic Regression, Naive Bayes, and Random Forest, the system analyzes the textual content of messages to make the classification decision.

---

## Dataset

The project uses the publicly available **SMS Spam Collection** dataset, which contains labeled messages classified as spam or ham. The dataset includes 5,574 SMS messages and the two categories are labeled as:

- **Spam**: Unsolicited, often unwanted messages.
- **Ham**: Legitimate, non-spam messages.

The dataset is stored in the `data/` folder and is available in CSV format.

---

## Repository Structure

Here's an overview of the project structure:

```
sms-spam-detection/
│
├── data/
│   ├── spam.csv           # The SMS Spam Collection dataset
│
├── notebooks/
│   ├── spam_detection.ipynb # Jupyter notebook for model development
│
├── src/
│   ├── preprocessing.py   # Preprocessing and text cleaning functions
│   ├── feature_extraction.py # TF-IDF vectorization
│   ├── model_training.py   # Model training and evaluation code
│
├── requirements.txt       # Required Python libraries for the project
├── README.md              # Project documentation
└── LICENSE                # License file
```

- **`data/`**: Contains the dataset (`spam.csv`).
- **`notebooks/`**: Contains the Jupyter notebook where the machine learning experiments and results are presented.
- **`src/`**: Python scripts for data preprocessing, feature extraction, and model training.
- **`requirements.txt`**: Python dependencies required to run the project.

---

## Installation

Follow the steps below to set up the environment and run the code locally.

### Prerequisites

Ensure you have **Python 3.x** installed. You can also use **virtual environments** to isolate your dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/sanidhyasinha/sms-spam-detection.git
   cd sms-spam-detection
   ```

2. Create a virtual environment (optional):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. To run the Jupyter notebook for model development:
   ```bash
   jupyter notebook notebooks/spam_detection.ipynb
   ```

---

## Usage

### Loading the Dataset

You can load the dataset from the `data/` folder using Pandas:

```python
import pandas as pd
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df.head()
```

### Data Preprocessing

Before training the model, preprocess the data by cleaning the text and encoding the labels:

```python
# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

# Encode labels (ham = 0, spam = 1)
df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

# Check for missing values
df.isnull().sum()
```

### Text Vectorization

Convert the textual data into numerical format using **TF-IDF** vectorization:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['Message'])
```

### Model Training and Evaluation

The dataset is split into training and test sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, df['Label'], test_size=0.2, random_state=42)
```

Now, you can train different models and evaluate them.

---

## Model Building

### Logistic Regression

To train a Logistic Regression model:

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
```

### Naive Bayes

To train a Naive Bayes classifier:

```python
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
```

### Random Forest

To train a Random Forest classifier:

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
```

You can evaluate the models using the following metrics:

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = lr_model.predict(X_test)  # For any model

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## Results and Evaluation

After training the models, you can evaluate them based on metrics like:

- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**

For example:

```python
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## Challenges and Future Improvements

### Challenges

- **Imbalanced Dataset**: There might be a higher number of ham messages than spam messages, which could affect model performance. You can explore techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** or **class weighting** to handle this.
- **Noise in Text Data**: Handling noisy data (like special characters, numbers, etc.) can be a challenge for text classification tasks.

### Future Improvements

- **Deep Learning Models**: Implement models like **LSTM** or **BERT** for better accuracy.
- **Real-time Spam Detection**: Integrate the model into a real-time application or API to detect spam messages as they are received.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This **README** provides a comprehensive overview of the project, including installation, usage, and model evaluation, structured according to the GitHub repository's folders. Let me know if you need further adjustments or additional details!
