## 🛍️ E-Commerce: Customer Review Summarization Using Text Mining  
This project was developed as part of the WIH3001 Data Science Project course at FSKTM, Universiti Malaya. It focuses on improving the online shopping experience by summarizing customer reviews and predicting product recommendations through machine learning and text mining techniques. The system condenses lengthy reviews, classifies sentiments, and predicts recommendations based on review content, helping both customers and businesses make more informed decisions.

## Features
- **Review Summarization**: Condense lengthy customer reviews using extractive text mining (Luhn, TF-IDF, LexRank, LSA).
- **Sentiment Classification**: Classify reviews into **positive**, **negative**, and **constructive** categories using a balanced SVM model.
- **Product Recommendation Prediction**: Predict whether a customer will recommend a product based on their review.
- **Interactive Web App**: Users can input their own reviews and receive real-time summarization, sentiment analysis, and recommendations.
- **Visualization**: Display of review insights including word clouds, sentiment distribution, rating distribution, and correlations.

## Data Sources
- **Customer Reviews Dataset**: Real anonymized e-commerce reviews from Kaggle.
- **Features**: Includes review text, product metadata, ratings, and recommendation indicators.

## Tools Used
- **Streamlit**: For developing the interactive web application.
- **Scikit-learn**: For training machine learning models (SVM, Random Forest, etc.).
- **TextBlob**: For initial sentiment polarity scoring.
- **TF-IDF & Luhn**: For text summarization.
- **Pandas & NumPy**: For data processing and analysis.
- **Matplotlib & Seaborn**: For visualizing EDA results.

## Model Highlights
- **Best Summarization Technique**: Luhn (ROUGE: 0.8994, BLEU: 0.7668)
- **Best Sentiment Model**: Balanced SVM (F1-scores improved across underrepresented classes)
- **Recommendation Accuracy**: 85.84% using linear SVM

## Web Application
Access the web app here: [Review Analyzer](https://reviewanalyzer-6tj2ztn8dyywm29k76p5nx.streamlit.app/)

