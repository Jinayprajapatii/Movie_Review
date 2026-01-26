🎬 Movie Review Sentiment Analysis

This project uses Natural Language Processing (NLP) and machine learning to predict the sentiment of movie reviews as Positive or Negative based on textual data from the IMDb dataset.

📊 Project Objective:

To build a text classification model that analyzes movie reviews and accurately predicts audience sentiment, helping understand viewer opinions at scale.

🔍 Key Highlights:

Text Cleaning & Preprocessing:

Converted text to lowercase

Removed HTML tags using Regular Expressions

Removed URLs, punctuation, special characters, and extra spaces

Removed duplicate reviews

Applied stopword removal

Feature Extraction:

Used Bag of Words (CountVectorizer)

Applied TF-IDF Vectorization with unigrams, bigrams, and trigrams to capture contextual meaning

Data Splitting:

Split data into training and testing sets using train_test_split

Model Building:

Implemented Logistic Regression for binary text classification

Model Evaluation:

Evaluated performance using:

Accuracy Score

Precision

Recall

F1-Score

Confusion Matrix

🛠️ Tools & Technologies:

Python (Pandas, NumPy, Scikit-learn, Regex)

Jupyter Notebook

NLP Techniques (TF-IDF, Bag of Words)

Matplotlib & Seaborn for visualization

🎯 Target Column:

sentiment

0 = Negative

1 = Positive

📈 Outcome:

The Logistic Regression model trained on TF-IDF features achieved strong classification performance, demonstrating that proper text preprocessing and feature extraction significantly improve sentiment prediction accuracy.

 👩‍💻 Created by: Jinay Prajapati  
 📅 Project Date: 2026  
