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
        'timestamp': pd.date_range(start='2024-01-01', periods=len(tweets), freq='H'),
        'likes': np.random.randint(0, 1000, len(tweets)),
        'retweets': np.random.randint(0, 500, len(tweets)),
        'user_id': [f'user_{i}' for i in range(len(tweets))]
    })
    
    return df

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("📊 Social Media Sentiment Analysis Dashboard")
    st.markdown("""
    Analyze the sentiment of social media posts using multiple NLP techniques.
    Upload your own data or use sample data to get started.
    """)
    
    # Initialize analyzer
    analyzer = StreamlitSentimentAnalyzer()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Choose Data Source:",
        ["Sample Data", "Upload CSV", "Manual Input"]
    )
    
    df = None
    
    # Handle different data sources
    if data_source == "Sample Data":
        if st.sidebar.button("Generate Sample Data"):
            df = create_sample_data()
            st.success("Sample data generated successfully!")
            
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded file with {len(df)} rows")
            
    elif data_source == "Manual Input":
        st.sidebar.subheader("Enter Text for Analysis")
        user_text = st.sidebar.text_area("Enter your text here:", height=100)
        if user_text and st.sidebar.button("Analyze Text"):
            df = pd.DataFrame({
                'text': [user_text],
                'timestamp': [pd.Timestamp.now()],
                'likes': [0],
                'retweets': [0],
                'user_id': ['manual_user']
            })
    
    # Main analysis
    if df is not None:
        
        # Column selection for text analysis
        if len(df.columns) > 1:
            text_column = st.sidebar.selectbox(
                "Select Text Column:",
                [col for col in df.columns if df[col].dtype == 'object']
            )
        else:
            text_column = 'text'
        
        # Display raw data
        with st.expander("📋 Raw Data Preview"):
            st.dataframe(df.head())
            st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Perform analysis
        if st.button("🔍 Analyze Sentiment", type="primary"):
            with st.spinner("Analyzing sentiment..."):
                analyzed_df = analyzer.analyze_dataframe(df, text_column)
                st.session_state.analyzed_data = analyzed_df
        
        # Display results if analysis is complete
        if st.session_state.analyzed_data is not None:
            analyzed_df = st.session_state.analyzed_data
            
            # Key metrics
            st.header("📈 Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_posts = len(analyzed_df)
                st.metric("Total Posts", total_posts)
            
            with col2:
                avg_textblob = analyzed_df['textblob_polarity'].mean()
                st.metric("Avg TextBlob Score", f"{avg_textblob:.3f}")
            
            with col3:
                avg_vader = analyzed_df['vader_compound'].mean()
                st.metric("Avg VADER Score", f"{avg_vader:.3f}")
            
            with col4:
                if 'likes' in analyzed_df.columns:
                    total_engagement = analyzed_df['likes'].sum() + analyzed_df['retweets'].sum()
                    st.metric("Total Engagement", total_engagement)
            
            # Sentiment Distribution
            st.header("📊 Sentiment Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("TextBlob Sentiment Distribution")
                tb_counts = analyzed_df['textblob_sentiment'].value_counts()
                fig_tb = px.pie(values=tb_counts.values, names=tb_counts.index,
                               color_discrete_map={
                                   'positive': '#2E8B57',
                                   'negative': '#DC143C',
                                   'neutral': '#708090'
                               })
                st.plotly_chart(fig_tb, use_container_width=True)
            
            with col2:
                st.subheader("VADER Sentiment Distribution")
                vader_counts = analyzed_df['vader_sentiment'].value_counts()
                fig_vader = px.pie(values=vader_counts.values, names=vader_counts.index,
                                  color_discrete_map={
                                      'positive': '#2E8B57',
                                      'negative': '#DC143C',
                                      'neutral': '#708090'
                                  })
                st.plotly_chart(fig_vader, use_container_width=True)
            
            # Sentiment Scores Distribution
            st.subheader("Sentiment Scores Distribution")
            
            fig_scores = go.Figure()
            fig_scores.add_trace(go.Histogram(
                x=analyzed_df['textblob_polarity'],
                name='TextBlob',
                opacity=0.7,
                nbinsx=30
            ))
            fig_scores.add_trace(go.Histogram(
                x=analyzed_df['vader_compound'],
                name='VADER',
                opacity=0.7,
                nbinsx=30
            ))
            fig_scores.update_layout(
                title="Distribution of Sentiment Scores",
                xaxis_title="Sentiment Score",
                yaxis_title="Frequency",
                barmode='overlay'
            )
            st.plotly_chart(fig_scores, use_container_width=True)
            
            # Time-based analysis (if timestamp available)
            if 'timestamp' in analyzed_df.columns:
                st.subheader("📅 Sentiment Over Time")
                
                # Create time-based aggregation
                analyzed_df['date'] = pd.to_datetime(analyzed_df['timestamp']).dt.date
                daily_sentiment = analyzed_df.groupby(['date', 'textblob_sentiment']).size().unstack(fill_value=0)
                
                fig_time = px.area(daily_sentiment, title="Sentiment Trends Over Time")
                st.plotly_chart(fig_time, use_container_width=True)
            
            # Engagement Analysis (if engagement data available)
            if 'likes' in analyzed_df.columns and 'retweets' in analyzed_df.columns:
                st.subheader("👍 Engagement Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_likes = px.box(analyzed_df, x='textblob_sentiment', y='likes',
                                      title="Likes by Sentiment")
                    st.plotly_chart(fig_likes, use_container_width=True)
                
                with col2:
                    fig_retweets = px.box(analyzed_df, x='textblob_sentiment', y='retweets',
                                         title="Retweets by Sentiment")
                    st.plotly_chart(fig_retweets, use_container_width=True)
            
            # Word Clouds
            st.subheader("☁️ Word Clouds by Sentiment")
            
            sentiments = ['positive', 'negative', 'neutral']
            cols = st.columns(3)
            
            for i, sentiment in enumerate(sentiments):
                with cols[i]:
                    sentiment_text = analyzed_df[
                        analyzed_df['textblob_sentiment'] == sentiment
                    ]['cleaned_text'].str.cat(sep=' ')
                    
                    if sentiment_text:
                        wordcloud = WordCloud(width=300, height=200,
                                            background_color='white',
                                            max_words=50).generate(sentiment_text)
                        
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.set_title(f'{sentiment.title()} Words')
                        ax.axis('off')
                        st.pyplot(fig)
            
            # Detailed Results Table
            st.subheader("📋 Detailed Results")
            
            # Add filters
            sentiment_filter = st.multiselect(
                "Filter by Sentiment:",
                options=analyzed_df['textblob_sentiment'].unique(),
                default=analyzed_df['textblob_sentiment'].unique()
            )
            
            filtered_df = analyzed_df[analyzed_df['textblob_sentiment'].isin(sentiment_filter)]
            
            # Display columns selection
            display_columns = st.multiselect(
                "Select Columns to Display:",
                options=analyzed_df.columns.tolist(),
                default=[text_column, 'textblob_sentiment', 'vader_sentiment', 
                        'textblob_polarity', 'vader_compound']
            )
            
            st.dataframe(filtered_df[display_columns])
            
            # Download results
            st.subheader("💾 Download Results")
            
            csv = analyzed_df.to_csv(index=False)
            st.download_button(
                label="Download Analysis Results as CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )
            
            # Model Comparison
            st.subheader("⚖️ Method Comparison")
            
            # Agreement between methods
            agreement = (analyzed_df['textblob_sentiment'] == analyzed_df['vader_sentiment']).mean()
            st.metric("TextBlob vs VADER Agreement", f"{agreement:.2%}")
            
            # Correlation between scores
            correlation = analyzed_df['textblob_polarity'].corr(analyzed_df['vader_compound'])
            st.metric("Score Correlation", f"{correlation:.3f}")
            
            # Confusion matrix
            confusion_data = pd.crosstab(analyzed_df['textblob_sentiment'], 
                                       analyzed_df['vader_sentiment'])
            
            fig_confusion = px.imshow(confusion_data,
                                    text_auto=True,
                                    title="TextBlob vs VADER Confusion Matrix",
                                    labels=dict(x="VADER Prediction", 
                                              y="TextBlob Prediction"))
            st.plotly_chart(fig_confusion, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this app:** This dashboard provides comprehensive sentiment analysis 
    using multiple NLP techniques including TextBlob and VADER sentiment analyzers.
    
    **Tips for better results:**
    - Use clean, well-formatted text data
    - Ensure sufficient data volume for meaningful insights
    - Consider domain-specific context when interpreting results
    """)

if __name__ == "__main__":
    main()
