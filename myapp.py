import os
os.environ['NLTK_DATA'] = './'

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopwords filtering
import nltk
nltk.data.find('tokenizers/punkt')  # Check if punkt is available
import numpy as np
from wordcloud import WordCloud
from collections import Counter

def luhn_summarize(text, num_sentences=3):
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    import string
    from collections import Counter

    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english')) | set(string.punctuation)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    word_freq = Counter(filtered_words)
    max_freq = max(word_freq.values())
    word_freq = {word: freq / max_freq for word, freq in word_freq.items()}

    sentence_scores = []
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        score = sum(word_freq.get(word, 0) for word in sentence_words)
        sentence_scores.append((sentence, score))

    ranked_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    summary = ' '.join([ranked_sentences[i][0] for i in range(min(num_sentences, len(ranked_sentences)))])
    return summary

@st.cache_data
def load_dataset(file_path):
    return pd.read_csv(file_path)

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)
# Load necessary data and models
nltk.download('punkt')
dataset = pd.read_csv("cleaned_reviews.csv")
dataset2 = pd.read_csv("original_reviews.csv")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
svm_model_balanced = joblib.load("svm_model_balanced.pkl")
svm_model = joblib.load("svm_model.pkl")
dataset_tfidf=joblib.load("dataset_tfidf.pkl")
luhn_summary_model = joblib.load("luhn_summary.pkl")

# Helper functions
def classify_sentiment(review):
    # Ensure that the input features match the model's expectations
    tfidf_vec = vectorizer.transform([review])
    if tfidf_vec.shape[1] != svm_model_balanced.n_features_in_:
        raise ValueError(
            f"Model expects {svm_model_balanced.n_features_in_} features, but got {tfidf_vec.shape[1]} features."
        )
    return svm_model_balanced.predict(tfidf_vec)[0]

def predict_recommendation(review, sentiment):
    # Predict recommendation based on sentiment
    if sentiment == "Positive":
        return "Recommend"
    elif sentiment == "Negative":
        return "Do not recommend"
    else:  # Constructive
        tfidf_vec = vectorizer.transform([review])
        prediction = svm_model.predict(tfidf_vec)[0]
        return "Recommend" if prediction == 1 else "Do not recommend"

def summarize_with_luhn(review, num_sentences=3):
    # Use the loaded Luhn summarization model
    return luhn_summary_model(review, num_sentences)

def suggest_alternatives(review):
    # Transform the review into TF-IDF
    tfidf_vec = vectorizer.transform([review])

    # Compute cosine similarities
    similarities = cosine_similarity(tfidf_vec, dataset_tfidf)
    most_similar_idx = np.argmax(similarities)

    # Debug: Check if the similarity index maps correctly
    print(f"Most Similar Index: {most_similar_idx}")
    print(f"Similar Review: {dataset.iloc[most_similar_idx]['Cleaned_ReviewText']}")

    # Extract category
    category = dataset.iloc[most_similar_idx]['Class_Name']

    # Suggest alternatives
    alternatives = dataset[(dataset['Class_Name'] == category) &
                           (dataset['Rating'] >= 4) &
                           (dataset['Recommended_IND'] == 1)].sort_values(
        by=['P_Feedback_Count'], ascending=False).head(3)
    return category, alternatives

def identify_category(review):
    # Define keywords and their associated categories
    keyword_category_map = {
        "Blouses": ["blouse"],
        "Casual bottoms": ["bottoms"],
        "Chemises": ["chemise","cami"],
        "Dresses": ["dress"],
        "Fine gauge": ["gauge"],
        "Intimates": ["bra", "panties", "undergarments","bikini","nightgown"],
        "Jackets": ["jacket"],
        "Jeans": ["jeans"],
        "Knits": ["knit"],
        "Layering": ["layer"],
        "Legwear": ["legging","tights","socks"],
        "Lounge": ["romper","yoga"],
        "Outerwear": ["outerwear"],
        "Pants": ["pant"],
        "Shorts": ["short"],
        "Skirts": ["skirt"],
        "Sleep": ["sleep","pajamas","robe"],
        "Sweaters": ["sweater"],
        "Swim": ["swim","bathing","suit"],
        "Trend": ["trend"]
    }

    # Check for keywords in the review text
    review_lower = review.lower()
    for category, keywords in keyword_category_map.items():
        for keyword in keywords:
            if keyword in review_lower:
                return category

    # Fallback to similarity-based identification if no keyword matches
    tfidf_vec = vectorizer.transform([review])
    similarities = cosine_similarity(tfidf_vec, dataset_tfidf)
    most_similar_idx = np.argmax(similarities)
    return dataset.iloc[most_similar_idx]['Class_Name']

# Ensure preprocessing is consistent
def preprocess_text(text):
    import re
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# Apply preprocessing to both sets of reviews
recommend_yes_clean = dataset[dataset['Recommended_IND'] == 1]['Cleaned_ReviewText'].apply(preprocess_text)
recommend_no_clean = dataset[dataset['Recommended_IND'] == 0]['Cleaned_ReviewText'].apply(preprocess_text)

# Combine reviews into single strings
corpus_yes = ' '.join(recommend_yes_clean)
corpus_no = ' '.join(recommend_no_clean)

# Create frequency distributions
from collections import Counter
freq_yes = Counter(corpus_yes.split())
freq_no = Counter(corpus_no.split())

# Identify common and unique words
common_words = set(freq_yes.keys()).intersection(set(freq_no.keys()))
unique_yes_words = set(freq_yes.keys()) - common_words
unique_no_words = set(freq_no.keys()) - common_words

# Create corpora for unique and common words
unique_yes_corpus = ' '.join([word for word in corpus_yes.split() if word in unique_yes_words])
unique_no_corpus = ' '.join([word for word in corpus_no.split() if word in unique_no_words])
common_corpus = ' '.join([word for word in corpus_yes.split() if word in common_words])

# Word Cloud Generation Function
@st.cache_data
def create_wordcloud(corpus, colormap):
    return WordCloud(width=800, height=400, background_color='white', colormap=colormap, max_words=100).generate(corpus)

# Generate word clouds
wordcloud_yes = create_wordcloud(unique_yes_corpus, 'Greens')
wordcloud_no = create_wordcloud(unique_no_corpus, 'Reds')
wordcloud_common = create_wordcloud(common_corpus, 'viridis')

# Function to get unique words for a specific sentiment
@st.cache_data
def get_unique_words(data, target_sentiment):
    all_words = Counter(" ".join(data['Cleaned_ReviewText']).split())
    sentiment_words = Counter(" ".join(data[data['Sentiment'] == target_sentiment]['Cleaned_ReviewText']).split())

    # Subtract the words of other sentiments
    for sentiment in data['Sentiment'].unique():
        if sentiment != target_sentiment:
            other_words = Counter(" ".join(data[data['Sentiment'] == sentiment]['Cleaned_ReviewText']).split())
            sentiment_words.subtract(other_words)

    # Remove words with zero or negative count
    unique_words = {word: count for word, count in sentiment_words.items() if count > 0}
    return unique_words

# Function to plot a word cloud using unique words
@st.cache_data
def plot_unique_word_cloud(unique_words, title):
    wordcloud = WordCloud(
        width=800, height=400, background_color='white', colormap='viridis'
    ).generate_from_frequencies(unique_words)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16)
    return fig
    
# Streamlit App
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    <h1 class="centered-title">Reviewly ♡ </h1>
    """,
    unsafe_allow_html=True
)

# Set up the query parameter detection
if "menu" not in st.session_state:
    query_params = st.query_params
    st.session_state.menu = query_params.get("menu", ["Project Overview"])[0]
else:
    st.session_state.menu = st.query_params.get("menu", ["Project Overview"])[0]

# Assign menu from session state
menu = st.session_state.menu
# Sidebar menu
with st.sidebar:
    menu = option_menu("Main Menu", ["Project Overview", "Analyzer", "Documentation"],
                       icons=["house", "bar-chart","book"], menu_icon="menu-app-fill", default_index=0)

if menu == "Project Overview":
    st.markdown(
        """
        <style>
        .stApp { background-color: #EFE5D9; } 
        h1 { color: #7F5E33 !important; }
        h2 { color: #A87C44 !important; }
        h3 { color: #C59F70 !important; }

        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        '<h2 class>Welcome to Reviewly!</h2>', unsafe_allow_html=True)

    st.markdown(
        """
         **“Have you ever struggled to choose a product because of too many conflicting reviews?”**                                 
            Discover what others are saying about their favorite products. Dive into real customer reviews to make informed decisions and find inspiration for your next purchase.
        """
    )
    # Adding the first image
    image_url1 = 'https://i.pinimg.com/736x/ca/30/4d/ca304d9b6db07e9fad43a9c04fec0096.jpg'
    st.markdown(
        f'''

                        <div style="display: flex; justify-content: center;">
                            <div class="image-transition">
                                <img src="{image_url1}" alt="Image" style="max-width: 600px; height: 450px; width: 800px;">
                            </div>
                        </div>

                        ''',
        unsafe_allow_html=True
    )
    # Navigation Tabs
    tabs = st.tabs(["Project Overview","EDA", "Code"])
    # Read your code file into a variable
    with open('DSproject.py', 'r') as file:
        project_code = file.read()

    # Project Overview Tab
    with tabs[0]:
        st.markdown("## **Project Overview**")

        # Introduction Section
        st.markdown("### **Introduction**")
        st.write(
            "With the rapid growth of e-commerce, customer reviews have become crucial for helping buyers make informed decisions. "
            "However, the sheer volume and lengthy reviews which include positive and negative reviews can make it difficult for customers to sift through them efficiently. "
            "This project uses text mining to summarize reviews making them easier to read and understand before making decisions to buy the products."
        )

        st.markdown("#### **What is Text Mining?**")
        st.write(
            "Text mining transforms unstructured text into meaningful insights through computational and linguistic techniques. "
            "It involves analyzing large amounts of text to extract patterns, trends or information that may not be easily visible. "
            "In customer reviews, text mining automates summarization and aids sentiment analysis by classifying reviews as positive, negative and constructive using machine learning models."
        )

        # Problem Statement Section
        st.markdown("### **Problem Statement**")
        st.write(
            "1. Customer reviews often contain mixed sentiments, free-from text and are lengthy making it difficult to digest and extract the important details\n"
            "2. Existing sentiment classification tools often focus on positive or negative reviews, overlooking constructive criticism\n"
            "3. Customer having difficulty in making informed decision to identify the best product based on the sheer volume of reviews"
        )

        # Objectives Section
        st.markdown("### **Objective**")
        st.write(
            "1. To summarize customer reviews using extractive text mining techniques.\n"
            "2. To classify and analyze customer reviews into sentiment categories: positive, negative and constructive feedback.\n"
            "3. To predict if a customer will recommend the product based on the textual content of their reviews using machine learning models."
        )
    with tabs[1]:
        st.header("Exploratory Data Analysis")

        # Top-level metrics
        cust_reviews = len(dataset)
        positive_reviews = len(dataset[dataset["Sentiment"] == "Positive"])
        negative_reviews = len(dataset[dataset["Sentiment"] == "Negative"])
        constructive_reviews = len(dataset[dataset["Sentiment"] == "Constructive"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customer Reviews", cust_reviews)
        col2.metric("Positive Sentiments", positive_reviews)
        col3.metric("Constructive Sentiments", constructive_reviews)

        st.subheader("Dataset Viewer")
        rows = st.slider("Select number of rows to view", min_value=20, max_value=10000, value=20, step=5)
        st.dataframe(dataset2.head(rows))

        # Add an expander for feature information
        with st.expander("Feature Information"):
            st.write("""
            - **Clothing ID**: Integer categorical variable that refers to the specific piece being reviewed.
            - **Age**: Positive integer variable of the reviewer’s age.
            - **Title**: String variable for the title of the review.
            - **Review Text**: String variable for the review body.
            - **Rating**: Positive ordinal integer variable for the product score granted by the customer from 1 (Worst) to 5 (Best).
            - **Recommended IND**: Binary variable stating whether the customer recommends the product (1 = recommended, 0 = not recommended).
            - **Positive Feedback Count**: Positive integer documenting the number of other customers who found this review positive.
            - **Division Name**: Categorical name of the product's high-level division.
            - **Department Name**: Categorical name of the product's department.
            - **Class Name**: Categorical name of the product's class.
            """)

        st.header("Sentiment Categories Distribution")
        sentiment_counts = dataset['Sentiment'].value_counts()

        # Create the pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        sentiment_counts.plot.pie(
            autopct='%1.1f%%',  # Display percentage
            startangle=90,  # Rotate the pie chart
            colors=['skyblue', 'lightgreen', 'salmon'],  # Colors for the slices
            labels=sentiment_counts.index,  # Labels for each sentiment
            ax=ax
        )

        # Add a title
        ax.set_title('Sentiment Distribution')
        ax.set_ylabel('')  # Hide the y-axis label
        st.pyplot(fig)
        st.write("Positive reviews dominate making up nearly half of all feedback (48.4%)")

        # Unique Word Clouds for Sentiments
        st.header("Unique Words in Sentiment Categories")
        for sentiment in dataset['Sentiment'].unique():
            st.subheader(f"Unique Word Cloud for {sentiment} Feedback")
            unique_words = get_unique_words(dataset, sentiment)
            wordcloud_fig = plot_unique_word_cloud(unique_words, f"Unique Word Cloud for {sentiment} Feedback")
            st.pyplot(wordcloud_fig)

        # Histogram for Review Length Distribution
        st.header("Distribution of Review Length")
        # Calculate review lengths
        dataset['Review_Length'] = dataset['Cleaned_ReviewText'].apply(lambda x: len(x.split()))

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(dataset['Review_Length'], bins=20, kde=True, color='coral', ax=ax)
        ax.set_title('Distribution of Review Length')
        ax.set_xlabel('Number of Words in Review')  # Label for x-axis
        ax.set_ylabel('Frequency')  # Label for y-axis
        st.pyplot(fig)
        st.write("Review length shows consistency across ratings, with most reviews falling between 20 and 60 words")

        st.header("Correlation Heatmap of Numerical Features")

        # Select only numerical columns for correlation
        numeric_data = dataset.select_dtypes(include=[np.number])

        # Compute the correlation matrix
        corr = numeric_data.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True, ax=ax)
        ax.set_title('Correlation Heatmap of Numerical Features')
        st.pyplot(fig)
        st.write("Rating has strong positive relathionship with the Recommended_IND")


        st.subheader("")
        st.subheader("Insights on Products")
        chart_option = st.selectbox("Select Insight Chart", ["Top 20 Most Common Words in Review", "Highest Negative Reviews","Highest Positive Reviews", "Distribution of Ratings","Unique Words in Recommended Reviews (Yes)","Unique Words in Not Recommended Reviews (No)"])
        if chart_option == "Top 20 Most Common Words in Review":
            image_url12 = 'https://64.media.tumblr.com/e817a565c57b505d33f4987b4e9e8018/4f777b1b61b2bbe0-02/s1280x1920/c76d5694c5d35fdb470131aec821a4274c41bec9.pnj'
            st.image(image_url12, caption='')
        elif chart_option == "Highest Negative Reviews":
            negative_reviews = dataset[dataset['Sentiment'] == "Negative"]['Class_Name'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=negative_reviews.values, y=negative_reviews.index, ax=ax, palette="Reds_r")
            ax.set_title("Departments with Highest Negative Reviews")
            st.pyplot(fig)
        elif chart_option == "Highest Positive Reviews":
            positive_reviews = dataset[dataset['Sentiment'] == "Positive"]['Class_Name'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=positive_reviews.values, y=positive_reviews.index, ax=ax, palette="Blues_r")
            ax.set_title("Departments with Highest Positive Reviews")
            st.pyplot(fig)
        elif chart_option == "Distribution of Ratings":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='Rating', data=dataset, palette='viridis', ax=ax)
            ax.set_title('Distribution of Ratings')
            ax.set_xlabel('Rating')  # Label for x-axis
            ax.set_ylabel('Count')  # Label for y-axis
            st.pyplot(fig)
        elif chart_option == "Unique Words in Recommended Reviews (Yes)":
            st.subheader("Unique Words in Recommended Reviews (Yes)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud_yes, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        elif chart_option == "Unique Words in Not Recommended Reviews (No)":
            st.subheader("Unique Words in Not Recommended Reviews (No)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud_no, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        elif chart_option == "Common Words Across Reviews (Yes & No)":
            st.subheader("Common Words Across Reviews (Yes & No)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud_common, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    # Placeholder for Code Tab
    with tabs[2]:
        st.markdown("## **Code**")
        st.write("Code implementation details will be placed here.")
        st.code(project_code, language='python')

elif menu == "Analyzer":
    st.markdown(
        """
        <style>
        .stApp { background-color: #EFE5D9; } 
        h1 { color: #7F5E33 !important; }
        h2 { color: #A87C44 !important; }
        h3 { color: #C59F70 !important; }
         /* Apply pink to all headers */

        </style>
        """,
        unsafe_allow_html=True
    )
    # Adding the first image
    image_url1 = 'https://64.media.tumblr.com/76846fb7bca01fbef14a683b2cb251f8/539db1c86ced3c55-ea/s640x960/f067f6373496ddbf0cc9482cbfdb733f098c9c59.jpg'
    st.markdown(
        f'''
        <div style="display: flex; justify-content: center;">
            <div class="image-transition">
                <img src="{image_url1}" alt="Image" style="max-width: 600px; height: 300px; width: 550px;">
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    #Navigation Tabs
    tabs = st.tabs(["Customer Review","Analyzer","Monitor"])

    #Review Tab
    with tabs[0]:

        # Dropdown for selecting Class_Name
        class_names = dataset['Class_Name'].unique()
        selected_class = st.selectbox("Select a Product Class to View Feedback", class_names)

        # Dropdown for selecting Age_Group
        age_groups = sorted(dataset['age_group'].unique()) # Get unique age groups
        selected_age_group = st.selectbox("Select Your Age Group", age_groups)  # User selects an age group

        # Dropdown for selecting Rating
        rating = sorted(dataset['Rating'].unique())  # Get unique age groups
        selected_rate = st.selectbox("Select the Rating Number", rating)  # User selects an age group

        # Calculate total positive feedback count for the selected class
        total_positive_feedback = dataset.loc[
            (dataset['Class_Name'] == selected_class) & (dataset['age_group'] == selected_age_group) & (
                        dataset['Rating'] == selected_rate), 'P_Feedback_Count'
        ].sum()

        st.markdown(f"**Total Positive Feedback for {selected_class}: {total_positive_feedback}**")

        example_reviews = dataset.loc[
            (dataset['Class_Name'] == selected_class) &
            (dataset['age_group'] == selected_age_group) &
            (dataset['Rating'] == selected_rate) &
            (dataset['Recommended_IND'] == 1) &
            (dataset['P_Feedback_Count'] > 0),
            ['Clothing_ID', 'Review_Text', 'P_Feedback_Count']  # Include review and feedback count
        ].sort_values(by='P_Feedback_Count', ascending=False).head(3)  # Get top 3 reviews

        # Display up to 3 example reviews
        if not example_reviews.empty:
            st.markdown("### Example Reviews with Positive Feedback:")
            for index, row in example_reviews.head(3).iterrows():
                st.markdown(
                    f"""
                        <div style="padding: 10px; margin-bottom: 15px; border: 2px solid #D3D3D3; border-radius: 10px; background-color: #F9F9F9;">
                            <p><b>Clothing ID:</b> {row['Clothing_ID']}</p>
                            <p><b>Review:</b> {row['Review_Text']}</p>
                            <p style="font-weight: bold; color: #2E8B57;"><b>Positive Feedback Count:</b> 🌟 {row['P_Feedback_Count']}</p>
                        </div>
                        """,
                    unsafe_allow_html=True
                )
        else:
            st.write("No reviews found with positive feedback for the selected criteria.")

        st.header("Customer Reviews and Ratings")
        # Creating a grid layout for the images
        col1, col2 = st.columns(2)

        # Adding the first image to the first column
        with col1:
            image_url5 = 'https://64.media.tumblr.com/7a0ed257fb53e4a81e493a14c8b85665/a453ce99586b4eed-17/s1280x1920/92c2a8cbbee92d43e5e3521766e4af66aeac0277.pnj'
            st.image(image_url5, caption='')

        # Adding the second image to the second column
        with col2:
            image_url2 = 'https://64.media.tumblr.com/185f95282b29ac0b5e9db316f7c0e77a/a453ce99586b4eed-c1/s1280x1920/1bd6676392210c9754302c89e094cd1f4426ae74.pnj'
            st.image(image_url2, caption='')

        # Creating a grid layout for the images
        col3, col4 = st.columns(2)
        with col3:
            image_url3 = 'https://64.media.tumblr.com/1a6790190d45807554b074789c5ab12e/9ffc9ba930bf4ce4-3a/s1280x1920/7bf3f5cf4cd90a58fd6fca21c9543fe7e08bbb3e.pnj'
            st.image(image_url3, caption='')

            # Adding the second image to the second column
        with col4:
            image_url4 = 'https://64.media.tumblr.com/18d48475399eacb2d4215c6abf108131/9ffc9ba930bf4ce4-5b/s1280x1920/fa2beed798d148c214bb6985c520b986ae656382.pnj'
            st.image(image_url4, caption='')
    #Analyzer Tab
    with tabs[1]:
        st.header("Customer Review Analyzer")
        review_input = st.text_area("Enter a customer review:")

        if st.button("Analyze Review"):
            if review_input:
                try:
                    summary = summarize_with_luhn(review_input)
                    sentiment = classify_sentiment(review_input)
                    recommendation = predict_recommendation(review_input, sentiment)

                    # Define sentiment colors
                    sentiment_color = {
                        "Positive": "green",
                        "Negative": "red",
                        "Constructive": "#ED7014"
                    }

                    recommendation_color ={
                        "Do not recommend": "red",
                        "Recommend": "green"
                    }

                    st.write(f"**Summary:** {summary}")
                    st.markdown(
                        f"**Sentiment:** <span style='color:{sentiment_color[sentiment]};'>{sentiment}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"**Recommendation:** <span style='color:{recommendation_color[recommendation]};'>{recommendation}</span>",
                        unsafe_allow_html=True
                    )

                    if recommendation == "Do not recommend":
                        category = identify_category(review_input)
                        st.write(f"**Category Identified:** {category}")
                        alternatives = dataset[(dataset['Class_Name'] == category) &
                                               (dataset['Rating'] >= 4) &
                                               (dataset['Recommended_IND'] == 1)].sort_values(
                            by=['P_Feedback_Count'], ascending=False).head(3)

                        st.write("**Suggested Alternatives for Category:**")
                        if not alternatives.empty:
                            for _, row in alternatives.iterrows():
                                stars = "⭐" * int(row['Rating'])
                                st.markdown(
                                    f"- **Clothing ID:** {row['Clothing_ID']}<br>"
                                    f"  **Rating:** {stars} ({row['Rating']})<br>"
                                    f"  **Positive Feedback Count:** 🌟 {row['P_Feedback_Count']}",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.write("No alternatives found.")
                except ValueError as e:
                    st.error(f"Error: {e}")

    with tabs[2]:
        st.header("Monitor User Input")
        user_data = st.session_state.get("user_data", pd.DataFrame(columns=['Review', 'Sentiment', 'Recommendation', 'Category']))

        if review_input:
            sentiment = classify_sentiment(review_input)
            category, _ = suggest_alternatives(review_input)
            user_data = pd.concat([user_data, pd.DataFrame.from_records([{
                'Review': review_input,
                'Sentiment': sentiment,
                'Recommendation': recommendation,
                'Category': category
            }])], ignore_index=True)
            st.session_state["user_data"] = user_data

        st.write(user_data)

        if not user_data.empty:
            st.subheader("Sentiment Summary Chart")
            sentiment_summary = user_data['Sentiment'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=sentiment_summary.index, y=sentiment_summary.values, ax=ax, palette="viridis")
            ax.set_title("Summary of Sentiments from User Inputs")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            st.pyplot(fig)

elif menu == "Documentation":
    st.markdown(
        """
        <style>
        .stApp { background-color: #EFE5D9; } 
        h1 { color: #7F5E33 !important; }
        h2 { color: #A87C44 !important; }
        h3 { color: #C59F70 !important; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("📚 Documentation")

    st.subheader("Project Overview")
    st.write(
        """
        The **[Project Overview](?menu=Analyzer)** menu provides an overview of the project. It has:
        - Information about customer review and text mining.
        - Problem statement and project objectives to be achieved.
        - Explore data distribution and trends using various charts and metrics.
        - Code for doing the data modeling and evaluation.
        """
    )

    st.subheader("Analyzer")
    st.write(
        """
        The **[Analyzer](?menu=Analyzer)** section is divided into tabs:
        - **Customer Review Tab**: Provides an overview of positive feedback count with their review by selecting the product class,age group and rating.
        - **Analyzer Tab**: Enter a review to analyze sentiment, summary, and recommendations.
        - **Monitor Tab**: Review previously entered data and analyze sentiment trends.
        """
    )

