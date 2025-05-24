import sys
sys.path.append('C:/Users/Wlink/anaconda3/Lib/site-packages')

import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('./data/dataset.csv')  # Make sure this path is correct

# Drop unnamed columns
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])

# Focus on text column and drop NaNs
df = df[['text']].dropna()

# Initialize stopwords and tokenizer
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Create dummy sentiment labels for demonstration (0 = Negative, 1 = Positive)
np.random.seed(42)
df['sentiment'] = np.random.randint(0, 2, size=len(df))

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Interactive prediction function
def predict_sentiment(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    label = "Positive" if pred == 1 else "Negative"
    return label

# Simple interactive chatbot loop with debug prints
def chatbot():
    print("\n=== Sentiment Chatbot ===")
    print("Type a sentence to analyze sentiment.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            print("\nGoodbye!")
            break

        print(f"Debug: received input: {user_input}")  # Debug line

        if user_input.strip().lower() == 'exit':
            print("Goodbye!")
            break

        if not user_input.strip():
            print("Please enter some text.")
            continue

        sentiment = predict_sentiment(user_input)
        print(f"Bot: Predicted Sentiment -> {sentiment}", flush=True)

if __name__ == "__main__":
    chatbot()
