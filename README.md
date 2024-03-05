# BBC-classification
News(BBC) text classification project

# Introduction
This project aims to classify text data through machine learning methods.Natural Language Processing (NLP) technologies are used, including data preprocessing, feature extraction, choices of features and text classification using Support Vector Machines (SVM).

## Installation Guide

## Prerequisites
Before you begin, ensure you have met the following requirements:
Python 3.x installed

Before running the project code, please ensure that the following necessary Python packages are installed in your Linux environment:
- numpy
- nltk
- pandas
- scikit-learn
- textblob
- gensim
- scipy
- seaborn
- matplotlib
You can install these packages by running the following command:
pip install numpy nltk pandas scikit-learn textblob gensim scipy seaborn matplotlib

## NLTK Data Download
Before you can use the code, you need to download some extra data for the NLTK library.Run Python and enter the following command: 
import nltk
- nltk.download('punkt')
- nltk.download('wordnet')
- nltk.download('omw-1.4')
or download all nltk data directly # nltk.download('all')

## Dataset Preparation
Place your dataset in the specified directory (according to your code, this should be datasets_coursework1/bbc).Ensure that the structure of the dataset conforms to the expected format in the code.

## Run the Project
- Ensure that the dataset has been placed in the correct directory.
- Open a terminal and navigate to the project directory.
- Run the main script file to start the processing and classification process:
- python CMT316_BBCNews.py

## Output
The project outputs the performance evaluation results of the model, including accuracy, precision, recall, and F1 score. In addition, a heatmap of the confusion matrix is generated to visually display the model performance.
## Support
If you encounter any problems running the project, make sure that all dependencies are installed correctly and check that Python and its packages are the latest versions.If the problem persists, refer to the installation guide and data preparation steps in this README file.
