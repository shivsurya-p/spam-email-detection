import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham':0, 'spam':1})

x = data['message']
y = data['label']

vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(x)

model = MultinomialNB()

model.fit(x, y)

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model Saved")