import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

df = pd.read_csv("../data/spam_cleaned.csv")

# add text features
df["num_characters"] = df["text"].apply(len)
df["num_words"] = df["text"].apply(lambda x: len(nltk.word_tokenize(x)))
df["num_sentences"] = df["text"].apply(lambda x: len(nltk.sent_tokenize(x)))

print("overall stats:")
print(df[["num_characters", "num_words", "num_sentences"]].describe())

print("\nham stats:")
print(df[df["target"] == 0][["num_characters", "num_words", "num_sentences"]].describe())

print("\nspam stats:")
print(df[df["target"] == 1][["num_characters", "num_words", "num_sentences"]].describe())

# class distribution
plt.figure(figsize=(6, 4))
plt.pie(df["target"].value_counts(), labels=["ham", "spam"], autopct="%0.2f%%")
plt.title("Spam vs Ham Distribution")
plt.tight_layout()
plt.show()

# character distribution
plt.figure(figsize=(10, 4))
sns.histplot(df[df["target"] == 0]["num_characters"], color="blue", label="ham", kde=True)
sns.histplot(df[df["target"] == 1]["num_characters"], color="red", label="spam", kde=True)
plt.legend()
plt.title("Number of Characters — Ham vs Spam")
plt.tight_layout()
plt.show()

# word distribution
plt.figure(figsize=(10, 4))
sns.histplot(df[df["target"] == 0]["num_words"], color="blue", label="ham", kde=True)
sns.histplot(df[df["target"] == 1]["num_words"], color="red", label="spam", kde=True)
plt.legend()
plt.title("Number of Words — Ham vs Spam")
plt.tight_layout()
plt.show()

# correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df[["target", "num_characters", "num_words", "num_sentences"]].corr(),
            annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
