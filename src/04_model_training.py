import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               BaggingClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier)
from sklearn.metrics import accuracy_score, precision_score

df = pd.read_csv("../data/spam_processed.csv")

# vectorize text using TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
x = tfidf.fit_transform(df["transformed_text"]).toarray()
y = df["target"].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2)

print("train size:", x_train.shape)
print("test size:", x_test.shape)


def train_classifier(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision


# compare multiple classifiers
classifiers = {
    "Logistic Regression":    LogisticRegression(solver="liblinear", penalty="l1"),
    "SVC":                    SVC(kernel="sigmoid", gamma=1.0),
    "KNN":                    KNeighborsClassifier(),
    "Decision Tree":          DecisionTreeClassifier(max_depth=5),
    "Random Forest":          RandomForestClassifier(n_estimators=50, random_state=2),
    "AdaBoost":               AdaBoostClassifier(n_estimators=50, random_state=2),
    "Bagging":                BaggingClassifier(n_estimators=50, random_state=2),
    "Extra Trees":            ExtraTreesClassifier(n_estimators=50, random_state=2),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=50, random_state=2),
    "Multinomial NB":         MultinomialNB(),
    "Bernoulli NB":           BernoulliNB(),
}

accuracy_scores = []
precision_scores = []

for name, clf in classifiers.items():
    acc, prec = train_classifier(clf, x_train, y_train, x_test, y_test)
    accuracy_scores.append(acc)
    precision_scores.append(prec)

performance_df = pd.DataFrame({
    "Algorithm": classifiers.keys(),
    "Accuracy": accuracy_scores,
    "Precision": precision_scores
}).sort_values("Precision", ascending=False)

print("\nmodel comparison:")
print(performance_df.to_string(index=False))

# best model — Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred = mnb.predict(x_test)

print("\nbest model: Multinomial Naive Bayes")
print("accuracy: ", round(accuracy_score(y_test, y_pred), 4))
print("precision:", round(precision_score(y_test, y_pred), 4))

# save model and vectorizer separately
pickle.dump(tfidf, open("../data/vectorizer.pkl", "wb"))
pickle.dump(mnb, open("../data/model.pkl", "wb"))
print("\nmodel saved to data/model.pkl")
print("vectorizer saved to data/vectorizer.pkl")
