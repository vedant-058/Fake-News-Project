# import streamlit as st
# import joblib
# import numpy as np

# # Load the saved model and vectorizer
# print("Loading model and vectorizer...")
# try:
#     model = joblib.load('logistic_model.pkl')
#     vector = joblib.load('tfidf_vectorizer.pkl')
#     print("Model and vectorizer loaded successfully.")
# except FileNotFoundError as e:
#     print(f"Error: {e}")
#     raise

# # Streamlit app interface
# st.title('Fake News Detector')
# input_text = st.text_input('Enter news article')

# # Define prediction function
# def prediction(input_text):
#     print("Transforming input text for prediction...")
#     input_data = vector.transform([input_text])
#     print("Making prediction...")
#     prediction = model.predict(input_data)
#     print(f"Prediction result: {prediction[0]}")
#     return prediction[0]

# # Display results in the app
# if input_text:
#     print("Input text received. Running prediction...")
#     pred = prediction(input_text)
#     if pred == 1:
#         st.write('The News is Fake')
#         print("The news is fake.")
#     else:
#         st.write('The News Is Real')
#         print("The news is real.")
# import streamlit as st
# import joblib
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# # Define stemming function (same as used in training)
# ps = PorterStemmer()
# def stemming(content):
#     stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
#     stemmed_content = stemmed_content.lower()
#     stemmed_content = stemmed_content.split()
#     stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
#     stemmed_content = ' '.join(stemmed_content)
#     return stemmed_content

# # Load the saved model and vectorizer
# print("Loading model and vectorizer...")
# try:
#     model = joblib.load('logistic_model.pkl')
#     vector = joblib.load('tfidf_vectorizer.pkl')
#     print("Model and vectorizer loaded successfully.")
# except FileNotFoundError as e:
#     print(f"Error loading model or vectorizer: {e}")
#     raise

# # Streamlit app interface
# st.title('Fake News Detector')
# input_text = st.text_input('Enter news article')

# # Define prediction function with preprocessing
# def prediction(input_text):
#     # Apply the same stemming preprocessing as done during training
#     print(f"Raw input text: {input_text}")
#     preprocessed_text = stemming(input_text)
#     print(f"Preprocessed text: {preprocessed_text}")
    
#     # Vectorize the input text
#     input_data = vector.transform([preprocessed_text])
#     print(f"Vectorized input data: {input_data}")
    
#     # Make the prediction
#     prediction = model.predict(input_data)
#     print(f"Prediction result: {prediction[0]}")
#     return prediction[0]

# # Display results in the app
# if input_text:
#     print("Input text received. Running prediction...")
#     pred = prediction(input_text)
#     if pred == 1:
#         st.write('The News is Fake')
#         print("The news is fake.")
#     else:
#         st.write('The News Is Real')
#         print("The news is real.")










# import pandas as pd
# import numpy as np
# import re
# import joblib
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# print("Starting training script...")

# # Load data
# print("Loading data...")
# news_df = pd.read_csv('train.csv')
# news_df = news_df.fillna(' ')
# news_df['content'] = news_df['author'] + ' ' + news_df['title']
# X = news_df.drop('label', axis=1)
# y = news_df['label']
# print(f"Data loaded with {len(news_df)} rows.")

# # Define stemming function
# ps = PorterStemmer()
# def stemming(content):
#     stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
#     stemmed_content = stemmed_content.lower()
#     stemmed_content = stemmed_content.split()
#     stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
#     stemmed_content = ' '.join(stemmed_content)
#     return stemmed_content

# print("Applying stemming...")
# # Apply stemming function to content column
# news_df['content'] = news_df['content'].apply(stemming)

# # Vectorize data
# print("Vectorizing data...")
# X = news_df['content'].values
# y = news_df['label'].values
# vector = TfidfVectorizer()
# vector.fit(X)
# X = vector.transform(X)
# print(f"Data vectorized with {X.shape[0]} samples and {X.shape[1]} features.")

# # Save vectorizer
# joblib.dump(vector, 'tfidf_vectorizer.pkl')
# print("Vectorizer saved.")

# # Split data into train and test sets
# print("Splitting data...")
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
# print(f"Training data: {X_train.shape[0]} samples, Test data: {X_test.shape[0]} samples")

# # Fit logistic regression model
# print("Training model...")
# model = LogisticRegression()
# model.fit(X_train, Y_train)

# # Save model
# joblib.dump(model, 'logistic_model.pkl')
# print("Model saved.")

# # Print class distribution
# print("Class distribution in the training set:")
# unique, counts = np.unique(Y_train, return_counts=True)
# class_distribution = dict(zip(unique, counts))
# print(class_distribution)

# # Evaluate model
# print("Evaluating model...")
# y_train_pred = model.predict(X_train)
# print(f"Training accuracy: {accuracy_score(Y_train, y_train_pred)}")

# y_test_pred = model.predict(X_test)
# print(f"Test accuracy: {accuracy_score(Y_test, y_test_pred)}")

# # Print model weights
# feature_names = vector.get_feature_names_out()
# coef = model.coef_.flatten()
# top_positive_coefficients = np.argsort(coef)[-10:]
# top_negative_coefficients = np.argsort(coef)[:10]

# print("Top positive coefficients:")
# for i in top_positive_coefficients:
#     print(f"{feature_names[i]}: {coef[i]}")

# print("Top negative coefficients:")
# for i in top_negative_coefficients:
#     print(f"{feature_names[i]}: {coef[i]}")





import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function for text preprocessing (same as used in model training)
def preprocess_text(content):
    # Remove all non-alphabetical characters and convert text to lowercase
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    
    # Tokenize and remove stopwords
    words = content.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    
    # Join back the processed words into a single string
    return ' '.join(words)

# Load the pre-trained logistic regression model and TF-IDF vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    st.write("Loading model and vectorizer...")
    model = joblib.load('logistic_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    st.write("Model and vectorizer loaded successfully.")
    return model, vectorizer

# Function to make predictions
def predict_news(article, model, vectorizer):
    st.write(f"Raw input text: {article}")
    
    # Preprocess the article
    preprocessed_article = preprocess_text(article)
    st.write(f"Preprocessed text: {preprocessed_article}")
    
    # Transform the article using the loaded TF-IDF vectorizer
    input_vector = vectorizer.transform([preprocessed_article])
    st.write(f"Vectorized input data shape: {input_vector.shape}")
    
    # Use the model to predict
    prediction = model.predict(input_vector)
    st.write(f"Prediction result: {prediction[0]}")
    
    # Return the result
    if prediction[0] == 1:
        return "The news is fake."
    else:
        return "The news is real."

# Streamlit app interface
def main():
    # App title
    st.title("Fake News Detector")

    # Load the model and vectorizer
    model, vectorizer = load_model_and_vectorizer()

    # Get news article input from the user
    input_text = st.text_area("Enter news article")

    # Button to trigger prediction
    if st.button("Classify News"):
        if input_text:
            # Predict if the news is real or fake
            result = predict_news(input_text, model, vectorizer)
            # Output the result
            st.subheader(result)
        else:
            st.write("Please enter a news article to classify.")

if __name__ == "__main__":
    main()
