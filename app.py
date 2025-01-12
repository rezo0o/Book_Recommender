import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings
import os
from kaggle.api.kaggle_api_extended import KaggleApi
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Book Recommender", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stImage > img {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 3px;
    }
    </style>
    """, unsafe_allow_html=True)

# Add a title and description
st.title("📚 Book Recommendation System")
st.write("Welcome to the Book Recommender! Discover your next favorite book.")

def download_kaggle_dataset():
    """Download dataset from Kaggle if not present locally"""
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Check if files exist
        files_exist = all(
            os.path.exists(f'data/{file}') 
            for file in ['Books.csv', 'Ratings.csv', 'Users.csv']
        )
        
        if not files_exist:
            with st.spinner('Downloading dataset from Kaggle...'):
                # Get credentials from Streamlit secrets
                os.environ['KAGGLE_USERNAME'] = st.secrets['kaggle']['username']
                os.environ['KAGGLE_KEY'] = st.secrets['kaggle']['key']
                
                api = KaggleApi()
                api.authenticate()
                api.dataset_download_files(
                    'arashnic/book-recommendation-dataset',
                    path='data',
                    unzip=True
                )
                st.success('Dataset downloaded successfully!')
                
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        st.info("Please check your Kaggle credentials in Streamlit secrets.")
        st.stop()

def load_image_from_url(url):
    try:
        if pd.isna(url) or not url:
            return None
        
        # Transform old Amazon URLs to a modern format
        if 'images.amazon.com' in url:
            url = url.replace('http://', 'https://')
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
    except Exception as e:
        return None
    return None

def create_placeholder_image():
    img = Image.new('RGB', (150, 200), color='lightgray')
    return img

@st.cache_data
def load_and_process_data():
    """Load and process the dataset"""
    # First ensure data is downloaded
    download_kaggle_dataset()
    
    try:
        # Load data
        books = pd.read_csv('data/Books.csv')
        ratings = pd.read_csv('data/Ratings.csv')
        users = pd.read_csv('data/Users.csv')

        # Drop Age column
        if "Age" in users.columns:
            users.drop(columns=["Age"], inplace=True)

        # Drop rows with missing values in books
        books.dropna(inplace=True)

        # Drop ratings where Book-Rating == 0
        ratings = ratings[ratings["Book-Rating"] != 0]

        # Merge DataFrames
        merged_df = pd.merge(books, ratings, on="ISBN", how="inner")
        merged_df = pd.merge(merged_df, users, on="User-ID", how="inner")

        # Clean Year-Of-Publication
        merged_df["Year-Of-Publication"] = pd.to_numeric(
            merged_df["Year-Of-Publication"],
            errors='coerce'
        )

        # Replace weird years with NaN
        merged_df.loc[
            (merged_df["Year-Of-Publication"] < 1500) |
            (merged_df["Year-Of-Publication"] > 2025),
            "Year-Of-Publication"
        ] = np.nan

        # Drop rows where Year-Of-Publication is NaN
        merged_df.dropna(subset=["Year-Of-Publication"], inplace=True)

        # Filter out users with fewer than 5 ratings
        user_counts = merged_df["User-ID"].value_counts()
        active_users = user_counts[user_counts >= 5].index
        merged_df = merged_df[merged_df["User-ID"].isin(active_users)]

        # Filter out books with fewer than 5 ratings
        book_counts = merged_df["Book-Title"].value_counts()
        active_books = book_counts[book_counts >= 5].index
        merged_df = merged_df[merged_df["Book-Title"].isin(active_books)]

        return merged_df, books, ratings, users
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()

@st.cache_data
def get_popular_books(merged_df):
    # Calculate ratings statistics
    pop_stats = merged_df.groupby("Book-Title").agg({
        "Book-Rating": ["count", "mean"],
        "ISBN": "first",
        "Book-Author": "first",
        "Publisher": "first",
        "Image-URL-M": "first"
    }).reset_index()
    
    pop_stats.columns = ["Book-Title", "NumRatings", "AvgRating", "ISBN", "Book-Author", "Publisher", "Image-URL-M"]
    
    # Compute IMDB weighted rating
    C = pop_stats["AvgRating"].mean()
    m = pop_stats["NumRatings"].quantile(0.90)
    
    pop_stats["WeightedRating"] = (
        (pop_stats["NumRatings"] / (pop_stats["NumRatings"] + m) * pop_stats["AvgRating"]) +
        (m / (pop_stats["NumRatings"] + m) * C)
    )
    
    pop_stats.sort_values("WeightedRating", ascending=False, inplace=True)
    return pop_stats

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📈 Popular Books", "👥 Similar Books", "📖 Content Based"])

# Load the processed data
with st.spinner('Loading data...'):
    merged_df, books_df, ratings_df, users_df = load_and_process_data()

# Show basic stats in the sidebar
with st.sidebar:
    st.header("Dataset Info")
    st.write(f"Total Books: {len(merged_df['Book-Title'].unique())}")
    st.write(f"Total Ratings: {len(merged_df)}")
    st.write(f"Total Users: {len(merged_df['User-ID'].unique())}")

# Display popular books in the first tab
with tab1:
    st.header("Popular Books")
    st.write("Top books based on the IMDB weighted rating formula")
    popular_books = get_popular_books(merged_df)
    
    # Show top 10 books with images
    for i in range(min(10, len(popular_books))):
        book = popular_books.iloc[i]
        col1, col2 = st.columns([1, 3])
        
        with col1:
            img = load_image_from_url(book['Image-URL-M'])
            if img is not None:
                st.image(img, width=150)
            else:
                placeholder_img = create_placeholder_image()
                st.image(placeholder_img, caption="Cover not available", width=150)
        
        with col2:
            st.subheader(book['Book-Title'])
            st.write(f"**Author:** {book['Book-Author']}")
            st.write(f"**Weighted Rating:** {book['WeightedRating']:.2f}")
            st.write(f"**Average Rating:** {book['AvgRating']:.2f} (from {book['NumRatings']} ratings)")
            st.write(f"**Publisher:** {book['Publisher']}")
            st.write("---")

# Similar Books Tab (Item-based)
with tab2:
    st.header("Find Similar Books")
    
    # Get book list for selection
    book_list = merged_df['Book-Title'].unique()
    selected_book = st.selectbox("Select a book:", book_list)
    
    if selected_book:
        # Create smaller pivot table for the selected book
        similar_books = merged_df[merged_df['Book-Title'] == selected_book]
        similar_users = similar_books['User-ID'].unique()
        user_books = merged_df[merged_df['User-ID'].isin(similar_users)]
        
        # Create pivot table
        item_pivot = user_books.pivot_table(
            index="User-ID",
            columns="Book-Title",
            values="Book-Rating"
        ).fillna(0)
        
        # Compute similarity
        item_similarity = cosine_similarity(item_pivot.T)
        books_list = item_pivot.columns
        
        # Get recommendations
        idx = books_list.get_loc(selected_book)
        sim_scores = list(enumerate(item_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Get top 5
        
        st.subheader("Similar Books:")
        for i, score in sim_scores:
            book_title = books_list[i]
            book_data = merged_df[merged_df['Book-Title'] == book_title].iloc[0]
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                img = load_image_from_url(book_data['Image-URL-M'])
                if img is not None:
                    st.image(img, width=150)
                else:
                    placeholder_img = create_placeholder_image()
                    st.image(placeholder_img, caption="Cover not available", width=150)
            
            with col2:
                st.write(f"**{book_title}**")
                st.write(f"Author: {book_data['Book-Author']}")
                st.write(f"Similarity Score: {score:.2f}")
                st.write("---")

# Content Based Tab
with tab3:
    st.header("Content-Based Recommendations")
    
    # Create content features for selected books only
    unique_books = merged_df.drop_duplicates(subset=["Book-Title"]).copy()
    book_list = unique_books['Book-Title'].unique()
    selected_book = st.selectbox("Select a book:", book_list, key='content_select')
    
    if selected_book:
        # Create content features for similar books
        selected_book_data = unique_books[unique_books['Book-Title'] == selected_book]
        similar_books = unique_books[unique_books['Book-Author'] == selected_book_data['Book-Author'].iloc[0]]
        
        # Combine features
        similar_books["ContentFeatures"] = (
            similar_books["Book-Author"].fillna("") + " " +
            similar_books["Publisher"].fillna("")
        )
        
        # Clean text
        similar_books["ContentFeatures"] = similar_books["ContentFeatures"].apply(
            lambda x: re.sub(r"[^\w\s]", "", str(x)).strip().lower()
        )
        
        # Create TF-IDF vectors
        tfidf = TfidfVectorizer(stop_words="english")
        content_matrix = tfidf.fit_transform(similar_books["ContentFeatures"])
        
        # Compute similarity
        content_similarity = cosine_similarity(content_matrix)
        content_books_list = similar_books["Book-Title"].tolist()
        
        idx = content_books_list.index(selected_book)
        sim_scores = list(enumerate(content_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Get top 5
        
        st.subheader("Books with Similar Content:")
        for i, score in sim_scores:
            book_title = content_books_list[i]
            book_data = similar_books[similar_books['Book-Title'] == book_title].iloc[0]
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                img = load_image_from_url(book_data['Image-URL-M'])
                if img is not None:
                    st.image(img, width=150)
                else:
                    placeholder_img = create_placeholder_image()
                    st.image(placeholder_img, caption="Cover not available", width=150)
            
            with col2:
                st.write(f"**{book_title}**")
                st.write(f"Author: {book_data['Book-Author']}")
                st.write(f"Publisher: {book_data['Publisher']}")
                st.write(f"Similarity Score: {score:.2f}")
                st.write("---")