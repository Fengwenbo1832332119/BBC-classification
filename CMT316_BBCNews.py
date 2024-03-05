import os
import glob
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from textblob import TextBlob
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize preprocessing tools
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# or nltk.download('all')
# We need to download these dependencies in advance so that we can perform tokenization as well as lemmatization of the text.
# Load a list of stop words in English.
# Stop words refer to words that frequently appear in a text but are not very helpful in understanding its meaning.
stop_words = set(stopwords.words('english'))
# Initialize a word form restorer to restore words to their root form.
lemmatizer = WordNetLemmatizer()

# Remove special characters and numbers by using \W+|\d+ regular regression
# This function performs a series of processing on the text as described above, achieving the purpose of cleaning the data
def preprocess_text(text):
    text = re.sub(r'\W+|\d+', ' ', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    return ' '.join(cleaned_tokens)

# Specify the root directory where the dataset is located
dataset_root = 'C:/Users/c23059233/OneDrive - Cardiff University/Desktop/machine learning/datasets_coursework1/bbc'

# Stores a list of all text and its labels
texts = []
labels = []

# Iterate through each folder and file
for label in os.listdir(dataset_root):
    folder_path = os.path.join(dataset_root, label)
    if os.path.isdir(folder_path):
        for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
                cleaned_text = preprocess_text(text)
                texts.append(cleaned_text)
                labels.append(label)

# Creating a DataFrame
# making DataFrame file to csv file
df = pd.DataFrame({'label': labels, 'text': texts})
df.to_csv('C:/Users/c23059233/OneDrive - Cardiff University/Desktop/machine learning/datasets_coursework1/bbc/bbc_dataset_cleaned.csv', index=False)

# Initialize the count vectorizer
# The parameter max_features=1000 limits the model to only consider the 1000 words with the highest word frequency.
vectorizer = CountVectorizer(max_features=1000)

# Compute TF. TF is a sparse matrix representing absolute word frequencies
TF = vectorizer.fit_transform(df['text'])

# Initializing the TF-IDF Vectorizer
Tfid_vectorizer = TfidfVectorizer(max_features=1000)

# Calculate TF-IDF
Tfid = Tfid_vectorizer.fit_transform(df['text'])

# Use MaxAbsScaler to scale word frequency (TF) and TF-IDF features.
# MaxAbsScaler scales the data by dividing it by the maximum absolute value of each feature, making the absolute value of all features maximum
# This method is particularly useful for sparse data as it does not alter the sparsity of the data.
scaler_maxabs = MaxAbsScaler()

TF_scaled = scaler_maxabs.fit_transform(TF)
Tfid_scaled = scaler_maxabs.fit_transform(Tfid)

# performing the chi-square test on the TF and TF-IDF features
# So first separate them from the feature matrix
features_X_text = hstack([TF_scaled, Tfid_scaled])

# Apply chi-square test to select the best K features
# Select the 1000 best features
chi2_selector = SelectKBest(chi2, k=1000)
features_Y = df['label']
features_X_text_selected = chi2_selector.fit_transform(features_X_text, features_Y)

# Calculating emotional polarity
# This function uses the TextBlob library to calculate the sentiment polarity score of the text.
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# df['sentiment'] contains the sentiment polarity score for each document
df['sentiment'] = df['text'].apply(get_sentiment)

# Preparing text data
texts = [text.split() for text in df['text']]

# Creating dictionaries and corpora
# Using the Gensim library, create a vocabulary based on text data.
# Convert each document into a Bag of Words (BoW) model, which represents the frequency of each word in the document.
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Training the LDA model
# Train an LDA topic model on a corpus using the LdaModel from the Gensim library,
# specifying the extraction of 5 topics and using the previously created dictionary as a reference.
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)

# Get the subject distribution of a document
# The call returned a list of topic distributions for a document, with each topic corresponding to a probability value.
def get_document_topics(bow):
    topic_probs = lda.get_document_topics(bow, minimum_probability=0)
    topic_prob_dict = dict(topic_probs)
    return [topic_prob_dict.get(i, 0) for i in range(5)]

# df['topics'] contains the distribution of topics for each document
df['topics'] = [get_document_topics(dictionary.doc2bow(text)) for text in texts]

# Use MinMaxScaler to scale sentiment scores and topic distribution.
# MinMaxScaler scales features to a given minimum and maximum value, typically between 0 and 1.
scaler_minmax = MinMaxScaler()

# Sentiment scores and thematic distributions have been converted to dense matrices
sentiment_scaled = scaler_minmax.fit_transform(df['sentiment'].values.reshape(-1, 1))
topics_scaled = scaler_minmax.fit_transform(np.vstack(df['topics'].values))

# Convert these processed non-sparse features to sparse format
sentiment_sparse_scaled = csr_matrix(sentiment_scaled)
topics_sparse_scaled = csr_matrix(topics_scaled)

# Stacking TF, TF-IDF, Sentiment Scores and Theme Distributions into a Large Feature Matrix
features_X_scaled = hstack([features_X_text_selected, sentiment_sparse_scaled, topics_sparse_scaled])

# First divide the test set, which is 20% of the original dataset
# Random_state=42 ensures consistency in each segmentation result.
X_temp, X_test, Y_temp, Y_test = train_test_split(features_X_scaled, features_Y, test_size=0.2, random_state=42)

# Then, the development set is divided from the remaining data, here we take 25% as the development set
X_train, X_dev, Y_train, Y_dev = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

# Use the training set to tune the parameters of the SVM model
# Define the search space
# 'C': Candidate values for regularization parameters. A smaller C value specifies stronger regularization.
# 'kernel': The type of kernel function to be used by SVM. We consider linear, radial basis, and polynomial kernels here.
# 'gamma': The coefficient of the kernel function, which only affects the 'rbf' and 'poly' kernels Scale 'represents automatic adjustment, while' auto 'and other values are candidate values manually set.
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Note! this can be adapted and expanded as needed
    'kernel': ['linear', 'rbf', 'poly'],  # Includes linear, RBF and polynomial kernels
    'gamma': ['scale', 'auto', 0.1, 1, 10]  # For RBF and polynomial kernels, gamma is an important parameter
}
# SVC() is the SVM model to be optimized.
# CV=5 indicates a cross validation fold of 5.
# Scoring='accuracy 'specifies the scoring standard as accuracy.
# N_jobs=-1 indicates using all available CPU cores for parallel computing, accelerating the search process
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy',n_jobs=-1)
grid_search.fit(X_train, Y_train)

# print the optimal parameter combination found
# print the optimal cross validation accuracy under these parameters
print("Optimal parameters:", grid_search.best_params_)
print("Optimal Cross Validation Accuracy:", grid_search.best_score_)

# Retrain the model with optimal parameters and all training and development data
# X_train and Y_train are used here, which contain data from the training and development sets
best_params = grid_search.best_params_
svm_model = SVC(**best_params)
svm_model.fit(X_train, Y_train)

# Predictions on the test set using the trained model
# besides, using development set to evaluate the performance on development
y_pred = svm_model.predict(X_test)
y_dev_pred = svm_model.predict(X_dev)

# Calculate and print performance metrics on the test set
accuracy = accuracy_score(Y_test, y_pred)
macro_precision = precision_score(Y_test, y_pred, average='macro')
macro_recall = recall_score(Y_test, y_pred, average='macro')
macro_f1 = f1_score(Y_test, y_pred, average='macro')
report = classification_report(Y_test, y_pred)
# Calculate and print performance metrics on the development set
accuracy_dev = accuracy_score(Y_dev, y_dev_pred)
macro_precision_dev = precision_score(Y_dev, y_dev_pred, average='macro')
macro_recall_dev = recall_score(Y_dev, y_dev_pred, average='macro')
macro_f1_dev = f1_score(Y_dev, y_dev_pred, average='macro')
report_dev = classification_report(Y_dev, y_dev_pred)

# Print the performance
print(report_dev)
print("Development Set Accuracy:", accuracy_dev)
print("Development Set Precision:", macro_precision_dev)
print("Development Set Recall:", macro_recall_dev)
print("Development Set F1 Score:", macro_f1_dev)

print(report)
print("Test Set Accuracy:", accuracy)
print("Macro-averaged precision:", macro_precision)
print("Macro-average recall:", macro_recall)
print("macro-averaged F1:", macro_f1)

# Calculate the confusion matrix
conf_mat = confusion_matrix(Y_test, y_pred)
conf_mat_dev = confusion_matrix(Y_dev, y_dev_pred)

# Heatmap obfuscation matrices with Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt="d",
            xticklabels=np.unique(y_pred),
            yticklabels=np.unique(Y_test))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat_dev, annot=True, fmt="d",
            xticklabels=np.unique(y_dev_pred),
            yticklabels=np.unique(Y_dev))
plt.title("Confusion Matrix(Development set)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()