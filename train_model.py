import pandas as pd
import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# DOWNLOAD STOPWORDS
nltk.download('stopwords')

# LOAD DATASET
data = pd.read_csv("spam.csv", encoding='latin-1')

# KEEP REQUIRED COLUMNS
data = data[['v1', 'v2']]

# RENAME COLUMNS
data.columns = ['label', 'message']

# CONVERT LABELS
data['label'] = data['label'].map({'ham':0, 'spam':1})

# STEMMER
ps = PorterStemmer()

# CLEANING FUNCTION
def clean_text(text):

    # convert to lowercase
    text = text.lower()

    # remove symbols and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # split words
    words = text.split()

    # remove stopwords and stemming
    words = [
        ps.stem(word)
        for word in words
        if word not in stopwords.words('english')
    ]

    return " ".join(words)

# APPLY CLEANING
data['message'] = data['message'].apply(clean_text)

# INPUT OUTPUT
x = data['message']
y = data['label']

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1,2)
)

x = vectorizer.fit_transform(x)

# TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL
model = MultinomialNB(alpha=0.1)

# TRAIN
model.fit(x_train, y_train)

# PREDICT
pred = model.predict(x_test)

# ACCURACY
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

# SAVE FILES
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model Saved Successfully")