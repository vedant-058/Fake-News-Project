# import streamlit as st
# import numpy as np
# import re
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score


# Load data
# news_df = pd.read_csv('train.csv')
# news_df = news_df.fillna(' ')
# news_df['content'] = news_df['author'] + ' ' + news_df['title']
# X = news_df.drop('label', axis=1)
# y = news_df['label']

# # Define stemming function
# ps = PorterStemmer()
# def stemming(content):
#     stemmed_content = re.sub('[^a-zA-Z]',' ',content)
#     stemmed_content = stemmed_content.lower()
#     stemmed_content = stemmed_content.split()
#     stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
#     stemmed_content = ' '.join(stemmed_content)
#     return stemmed_content

# # Apply stemming function to content column
# news_df['content'] = news_df['content'].apply(stemming)

# # Vectorize data
# X = news_df['content'].values
# y = news_df['label'].values
# vector = TfidfVectorizer()
# vector.fit(X)
# X = vector.transform(X)

# # Split data into train and test sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# # Fit logistic regression model
# model = LogisticRegression()
# model.fit(X_train,Y_train)



# @st.cache_data
# def load_data():
#     news_df = pd.read_csv('train.csv')
#     news_df = news_df.fillna(' ')
#     news_df['content'] = news_df['author'] + ' ' + news_df['title']
#     return news_df

# @st.cache_data
# def train_model():
#     # Vectorize data
#     vector = TfidfVectorizer()
#     X = vector.fit_transform(news_df['content'].values)
#     y = news_df['label'].values
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
#     model = LogisticRegression()
#     model.fit(X_train, Y_train)
#     return vector, model

# news_df = load_data()
# vector, model = train_model()

# # website
# st.title('Fake News Detector')
# input_text = st.text_input('Enter news Article')

# def prediction(input_text):
#     input_data = vector.transform([input_text])
#     prediction = model.predict(input_data)
#     return prediction[0]

# if input_text:
#     pred = prediction(input_text)
#     if pred == 1:
#         st.write('The News is Fake')
#     else:
#         st.write('The News Is Real')

import streamlit as st
import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

# Load data
st.write("Loading data...")
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

st.write("Preprocessing data...")
# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
st.write("Vectorizing data...")
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
st.write("Splitting data into training and testing sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
st.write("Training the model...")
model = LogisticRegression()
model.fit(X_train, Y_train)

# Website
st.title('Fake News Detector')

# Adding informative content
st.write("""
### Welcome to the Fake News Detector
In an era where misinformation can spread rapidly, it's crucial to identify credible news sources. This tool leverages machine learning to help distinguish between real and fake news articles.

#### How It Works:
1. **Data Collection**: We utilize a dataset containing various news articles, labeled as real or fake.
2. **Preprocessing**: Articles are cleaned and stemmed to focus on essential words, removing common stopwords.
3. **Vectorization**: The content is transformed into numerical format using TF-IDF, allowing the model to understand the importance of each word.
4. **Model Training**: We train a Logistic Regression model on this processed data to make predictions on new articles.

#### Instructions:
- Enter a news article in the text box below, and our model will analyze it for you.
""")

input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    st.write("Processing your input...")
    pred = prediction(input_text)
    if pred == 1:
        st.write('### The News is Fake')
    else:
        st.write('### The News Is Real')
