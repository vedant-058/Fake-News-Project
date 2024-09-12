import joblib
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Start process
print("Starting the training process...")

# Load data
try:
    news_df = pd.read_csv('train.csv')
    print("Loaded dataset successfully.")
except FileNotFoundError:
    print("Error: train.csv file not found!")
    raise

# Fill NaN values and combine 'author' and 'title' into 'content'
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
print("Missing values filled and content column created.")

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)
print("Stemming applied to content.")

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values

vector = TfidfVectorizer()
X = vector.fit_transform(X)
print("Data vectorized using TfidfVectorizer.")

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
print("Data split into training and testing sets.")

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)
print("Logistic regression model trained successfully.")

# Save the model and the vectorizer
try:
    joblib.dump(model, 'logistic_model.pkl')
    joblib.dump(vector, 'tfidf_vectorizer.pkl')
    print("Model and vectorizer saved to logistic_model.pkl and tfidf_vectorizer.pkl.")
except Exception as e:
    print(f"Error saving model or vectorizer: {e}")
    raise

print("Training process completed successfully.")
