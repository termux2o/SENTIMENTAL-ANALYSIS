import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()   # load .env file

file_path = os.getenv("DATA_PATH")
print(file_path)
# Load the dataset locally
df = pd.read_csv(file_path, encoding='latin-1', header=None)

# Rename the columns for easier understanding
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Drop unnecessary columns
df.drop(columns=['id', 'date', 'query', 'user'], inplace=True)

import nltk
nltk.download('stopwords')

# Step 3: Preprocess the text data
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]  # Stemming and removing stopwords
    return ' '.join(text)
# sample=df['text'].iloc[:1000]
df['clean_text'] = df['text'].apply(preprocess_text)

# Display the cleaned text
print("\nCleaned data preview:")
print(df[['text', 'clean_text']].head())

# Step 4: Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['clean_text'])

print("\nTF-IDF matrix shape:", X.shape)

#============== Model Training=============

y = df['sentiment']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    # "Random Forest": RandomForestClassifier(n_estimators=5)
}


from sklearn.metrics import accuracy_score, precision_score

results = []
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')

    results.append((name, acc, prec))

    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Precision: {prec:.4f}")

    # smart selection based on accuracy + precision
    score = (acc * 0.6) + (prec * 0.4)

    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name


print("\n==============================")
print(f" Best Model Selected: {best_model_name}")
print(f"Score: {best_score:.4f}")
print("==============================")
def predict_sentiment(text):
    text_clean = text.lower()
    text_vec = vectorizer.transform([text_clean])
    pred = best_model.predict(text_vec)[0]

    if pred == 0:
        return "Negative 😡"
    elif pred == 1:
        return "Neutral 😐"
    else:
        return "Positive 😊"
    
import pandas as pd

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision"])
print(results_df.sort_values(by="Accuracy", ascending=False))

print(predict_sentiment("I love this product"))
print(predict_sentiment("Worst service ever"))


import joblib
import os

# save best model
joblib.dump(best_model, "../model/model.pkl")

# save vectorizer also (VERY IMPORTANT)
joblib.dump(vectorizer, "../model/vectorizer.pkl")

print("✅ Model and vectorizer saved successfully")


