import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../data/spam.csv", encoding="utf-8", encoding_errors="ignore")

print("Shape:", df.shape)
print(df.head())

# drop unnamed columns
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)

# rename columns
df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)

# encode target: ham=0, spam=1
encoder = LabelEncoder()
df["target"] = encoder.fit_transform(df["target"])

# check missing values
print("missing values:", df.isnull().sum())
print("duplicates:", df.duplicated().sum())

# remove duplicates
df = df.drop_duplicates(keep="first")
print("shape after removing duplicates:", df.shape)

print(df["target"].value_counts())

df.to_csv("../data/spam_cleaned.csv", index=False)
print("saved to data/spam_cleaned.csv")
