import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import re

# Set page configuration
st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None

class StreamlitSentimentAnalyzer:
    """Streamlit-optimized sentiment analyzer"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def clean_text(self, text):
        """Clean text for analysis"""
        if pd.isna(text):
            return ""
        
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text.strip()
    
    def get_textblob_sentiment(self, text):
        """Get TextBlob sentiment"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def get_vader_sentiment(self, text):
        """Get VADER sentiment"""
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_dataframe(self, df, text_column):
        """Analyze sentiment for entire dataframe"""
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Get sentiments
        df['textblob_sentiment'] = df['cleaned_text'].apply(self.get_textblob_sentiment)
        df['vader_sentiment'] = df['cleaned_text'].apply(self.get_vader_sentiment)
        
        # Get numerical scores
        df['textblob_polarity'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['vader_compound'] = df['cleaned_text'].apply(lambda x: self.vader_analyzer.polarity_scores(x)['compound'])
        
        return df

def create_sample_data():
    """Create sample data for demonstration"""
    
    tweets = [
        "I love this new product! Amazing quality and great service!",
        "Terrible service! Never ordering from here again!",
        "Just had lunch. The food was okay, nothing special.",
        "What a beautiful day! Feeling grateful and happy!",
        "This product is completely useless! Waste of money!",
        "It's raining today. Need to carry an umbrella.",
        "Just had the best meal ever! Highly recommend this restaurant!",
        "Stuck in traffic for 2 hours! This is so frustrating!",
        "The meeting is scheduled for 3 PM tomorrow.",
        "Excited about the new movie release! Can't wait to watch it!"
    ] * 10  # Multiply for more data
    
    # Add some variation
    np.random.seed(42)
    additional_text = [f" #{np.random.randint(1000, 9999)}" for _ in tweets]
    tweets = [tweet + add for tweet, add in zip(tweets, additional_text)]
    
    df = pd.DataFrame({
        'text': tweets,
        'timestamp': pd.date_range(start='2024-01-01', periods=len
