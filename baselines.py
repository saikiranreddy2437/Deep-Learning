import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
import joblib

train = pd.read_csv("data/train.tsv", sep="\t")
val = pd.read_csv("data/val.tsv", sep="\t")

def build_input(df):
    return (df['turn1'] + " [SEP] " + df['turn2'] + " [SEP] " + df['turn3']).values

X_train = build_input(train)
X_val = build_input(val)

y_train = train['label']
y_val = val['label']

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True)
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)

print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_vec, y_train)
lr_pred = lr.predict(X_val_vec)
print("LR Macro F1:", f1_score(y_val, lr_pred, average='macro'))
print(classification_report(y_val, lr_pred))

joblib.dump(lr, "outputs/logreg_model.joblib")

print("\nTraining Linear SVM...")
svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_pred = svm.predict(X_val_vec)
print("SVM Macro F1:", f1_score(y_val, svm_pred, average='macro'))
print(classification_report(y_val, svm_pred))

joblib.dump(svm, "outputs/svm_model.joblib")
