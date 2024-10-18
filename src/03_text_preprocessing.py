import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

ps = PorterStemmer()

df = pd.read_csv("../data/spam_cleaned.csv")


def transform_text(text):
    # lowercase
    text = text.lower()

    # tokenize
    text = nltk.word_tokenize(text)

    # remove non alphanumeric
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # remove stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    # stemming — convert all word forms to root form
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


df["transformed_text"] = df["text"].apply(transform_text)
print(df[["text", "transformed_text"]].head())

# most common spam words
spam_corpus = []
for msg in df[df["target"] == 1]["transformed_text"].tolist():
    for word in msg.split():
        spam_corpus.append(word)

ham_corpus = []
for msg in df[df["target"] == 0]["transformed_text"].tolist():
    for word in msg.split():
        ham_corpus.append(word)

print("\ntop 20 spam words:")
print(pd.DataFrame(Counter(spam_corpus).most_common(20)))

print("\ntop 20 ham words:")
print(pd.DataFrame(Counter(ham_corpus).most_common(20)))

df.to_csv("../data/spam_processed.csv", index=False)
print("\nsaved to data/spam_processed.csv")
