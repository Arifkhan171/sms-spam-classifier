# SMS / Email Spam Classifier

A natural language processing project that classifies SMS and email messages as spam or not spam using Multinomial Naive Bayes — achieving 97% accuracy and 100% precision on test data.

---

## About

This project processes raw text messages, cleans and transforms them using NLP techniques, and trains a classifier to detect spam. A Streamlit web app allows live message classification.

---

## Project Structure

```
sms-spam-classifier/
│
├── src/
│   ├── 01_data_cleaning.py       # remove duplicates, encode labels
│   ├── 02_eda.py                 # analyze spam vs ham patterns
│   ├── 03_text_preprocessing.py  # tokenize, remove stopwords, stem
│   └── 04_model_training.py      # train and compare classifiers
│
├── data/
│   └── spam.csv                  # raw dataset (5572 messages)
│
├── app.py                        # Streamlit web app
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Steps

### 1. Data Cleaning

- Dropped 3 unnamed columns
- Renamed columns to `target` and `text`
- Encoded labels: ham = 0, spam = 1
- Removed 403 duplicate rows
- Final dataset: 5169 messages

### 2. Exploratory Data Analysis

Key findings from EDA:

- Dataset is imbalanced: 87% ham, 13% spam
- Spam messages are significantly longer than ham
- Spam messages have more words on average
- High correlation between number of characters and spam label

| Metric | Ham | Spam |
|--------|-----|------|
| Avg characters | 71 | 138 |
| Avg words | 15 | 28 |
| Avg sentences | 1.9 | 3.4 |

### 3. Text Preprocessing

Each message goes through 5 steps:

1. Lowercase conversion
2. Tokenization
3. Remove non-alphanumeric characters
4. Remove stopwords and punctuation
5. Stemming — convert words to root form (running → run)

```python
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    # remove special characters
    # remove stopwords
    # apply stemming
    return " ".join(y)
```

### 4. Model Training

Text vectorized using TF-IDF with 3000 features. Compared 11 classifiers:

| Algorithm | Accuracy | Precision |
|-----------|----------|-----------|
| Multinomial NB | 0.9709 | 1.0000 |
| Bernoulli NB | 0.9834 | 0.9858 |
| Extra Trees | 0.9767 | 0.9740 |
| Random Forest | 0.9767 | 0.9740 |
| SVC | 0.9767 | 0.9740 |
| Logistic Regression | 0.9554 | 0.9231 |

Best model: **Multinomial Naive Bayes**
- Accuracy: 97.09%
- Precision: 100% (zero false positives)

Precision was prioritized over accuracy — a false positive means a real message gets marked as spam, which is worse than missing spam.

---

## How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/Arifkhan171/sms-spam-classifier.git
cd sms-spam-classifier
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run pipeline in order:**
```bash
python src/01_data_cleaning.py
python src/02_eda.py
python src/03_text_preprocessing.py
python src/04_model_training.py
```

**4. Launch web app:**
```bash
streamlit run app.py
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## Dataset

- Source: Kaggle — SMS Spam Collection Dataset
- Size: 5572 messages
- Columns: label (ham/spam), message text

---

## Author

**Arif Khan**
Final Year CS Student | University of Loralai | ML / Deep Learning / Agentic AI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/arif-khan-71a711376)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/Arifkhan171)
