# Configuration file for Social Media Sentiment Analysis Project
# This file contains all configurable parameters and settings

import os
from pathlib import Path

class Config:
    """Configuration class for sentiment analysis project"""
    
    # =====================================
    # PROJECT SETTINGS
    # =====================================
    
    PROJECT_NAME = "Social Media Sentiment Analysis"
    VERSION = "1.0.0"
    AUTHOR = "Your Name"
    
    # Directory structure
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    NOTEBOOKS_DIR = BASE_DIR / "notebooks"
    RESULTS_DIR = DATA_DIR / "results"
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR, RESULTS_DIR]:
        directory.mkdir(exist_ok=True)
    
    # =====================================
    # DATA SETTINGS
    # =====================================
    
    # Sample data parameters
    SAMPLE_DATA_SIZE = 1000
    RANDOM_SEED = 42
    
    # Data file paths
    RAW_DATA_FILE = DATA_DIR / "raw_social_media_data.csv"
    PROCESSED_DATA_FILE = DATA_DIR / "processed_data.csv"
    RESULTS_FILE = RESULTS_DIR / "sentiment_analysis_results.csv"
    
    # Text processing parameters
    MIN_TEXT_LENGTH = 3
    MAX_TEXT_LENGTH = 500
    
    # =====================================
    # MODEL SETTINGS
    # =====================================
    
    # Machine Learning parameters
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    CV_FOLDS = 5
    
    # Feature extraction
    MAX_FEATURES = 5000
    MIN_DF = 2
    MAX_DF = 0.95
    NGRAM_RANGE = (1, 2)
    
    # Model parameters
    LOGISTIC_REGRESSION_PARAMS = {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': RANDOM_SEED
    }
    
    NAIVE_BAYES_PARAMS = {
        'alpha': 1.0
    }
    
    SVM_PARAMS = {
        'C': 1.0,
        'kernel': 'linear',
        'random_state': RANDOM_SEED
    }
    
    # Model file paths
    TRAINED_MODELS_FILE = MODELS_DIR / "trained_models.pkl"
    VECTORIZER_FILE = MODELS_DIR / "vectorizer.pkl"
    
    # =====================================
    # SENTIMENT ANALYSIS SETTINGS
    # =====================================
    
    # TextBlob thresholds
    TEXTBLOB_POSITIVE_THRESHOLD = 0.1
    TEXTBLOB_NEGATIVE_THRESHOLD = -0.1
    
    # VADER thresholds
    VADER_POSITIVE_THRESHOLD = 0.05
    VADER_NEGATIVE_THRESHOLD = -0.05
    
    # Sentiment labels
    SENTIMENT_LABELS = ['negative', 'neutral', 'positive']
    SENTIMENT_COLORS = {
        'positive': '#2E8B57',  # Sea Green
        'negative': '#DC143C',  # Crimson
        'neutral': '#708090'    # Slate Gray
    }
    
    # =====================================
    # TEXT PREPROCESSING SETTINGS
    # =====================================
    
    # Regular expressions for cleaning
    URL_PATTERN = r'http\S+|www\S+|https\S+'
    MENTION_PATTERN = r'@\w+'
    HASHTAG_PATTERN = r'#\w+'
    SPECIAL_CHARS_PATTERN = r'[^a-zA-Z\s]'
    
    # Stop words and language settings
    LANGUAGE = 'english'
    REMOVE_STOPWORDS = True
    APPLY_LEMMATIZATION = True
    
    # =====================================
    # VISUALIZATION SETTINGS
    # =====================================
    
    # Plot settings
    FIGURE_SIZE = (12, 8)
    DPI = 300
    STYLE = 'seaborn-v0_8'
    
    # Color palettes
    MAIN_PALETTE = "husl"
    SEQUENTIAL_PALETTE = "viridis"
    DIVERGING_PALETTE = "RdYlBu"
    
    # Word cloud settings
    WORDCLOUD_WIDTH = 800
    WORDCLOUD_HEIGHT = 400
    WORDCLOUD_MAX_WORDS = 100
    WORDCLOUD_BACKGROUND = 'white'
    
    # =====================================
    # API SETTINGS (for future use)
    # =====================================
    
    # Twitter API (replace with your credentials)
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', '')
    
    # Reddit API (replace with your credentials)
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'SentimentAnalysis/1.0')
    
    # =====================================
    # STREAMLIT SETTINGS
    # =====================================
    
    # App configuration
    STREAMLIT_PAGE_TITLE = "Social Media Sentiment Analysis"
    STREAMLIT_PAGE_ICON = "📊"
    STREAMLIT_LAYOUT = "wide"
    
    # UI settings
    SIDEBAR_WIDTH = 300
    MAX_UPLOAD_SIZE = 200  # MB
    
    # =====================================
    # LOGGING SETTINGS
    # =====================================
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = BASE_DIR / 'sentiment_analysis.log'
    
    # =====================================
    # PERFORMANCE SETTINGS
    # =====================================
    
    # Processing limits
    MAX_BATCH_SIZE = 1000
    MAX_TEXT_SAMPLES = 10000
    
    # Parallel processing
    N_JOBS = -1  # Use all available cores
    
    # Memory management
    CHUNK_SIZE = 1000
    
    # =====================================
    # VALIDATION SETTINGS
    # =====================================
    
    # Model validation thresholds
    MIN_ACCURACY_THRESHOLD = 0.60
    MIN_F1_THRESHOLD = 0.55
    
    # Data quality thresholds
    MAX_MISSING_DATA_PERCENT = 10
    MIN_SAMPLES_PER_CLASS = 50
    
    # =====================================
    # EXPORT SETTINGS
    # =====================================
    
    # Output formats
    SUPPORTED_EXPORT_FORMATS = ['csv', 'json', 'xlsx', 'parquet']
    DEFAULT_EXPORT_FORMAT = 'csv'
    
    # Report settings
    INCLUDE_VISUALIZATIONS_IN_REPORT = True
    REPORT_TEMPLATE_PATH = BASE_DIR / "templates" / "report_template.html"

class DataSourceConfig:
    """Configuration for different data sources"""
    
    # Sample data configuration
    SAMPLE_DATA = {
        'positive_samples': 300,
        'negative_samples': 300,
        'neutral_samples': 300,
        'add_noise': True,
        'noise_level': 0.1
    }
    
    # CSV upload configuration
    CSV_CONFIG = {
        'text_column': 'text',
        'timestamp_column': 'timestamp',
        'label_column': 'sentiment',
        'encoding': 'utf-8',
        'delimiter': ','
    }
    
    # Twitter data configuration
    TWITTER_CONFIG = {
        'max_tweets': 1000,
        'lang': 'en',
        'result_type': 'recent',
        'include_retweets': False
    }
    
    # Reddit data configuration
    REDDIT_CONFIG = {
        'subreddits': ['politics', 'worldnews', 'technology'],
        'max_posts': 500,
        'time_filter': 'week'
    }

class ModelConfig:
    """Configuration for different models"""
    
    # Ensemble model settings
    ENSEMBLE_METHODS = ['voting', 'stacking', 'bagging']
    DEFAULT_ENSEMBLE_METHOD = 'voting'
    
    # Deep learning settings (for future implementation)
    DEEP_LEARNING_CONFIG = {
        'model_name': 'bert-base-uncased',
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 3
    }
    
    # Custom model paths
    CUSTOM_MODEL_PATHS = {
        'bert': MODELS_DIR / 'bert_sentiment.pth',
        'lstm': MODELS_DIR / 'lstm_sentiment.h5',
        'transformer': MODELS_DIR / 'transformer_sentiment.bin'
    }

# Utility functions for configuration
def get_model_config(model_name: str) -> dict:
    """Get configuration for a specific model"""
    config_map = {
        'logistic_regression': Config.LOGISTIC_REGRESSION_PARAMS,
        'naive_bayes': Config.NAIVE_BAYES_PARAMS,
        'svm': Config.SVM_PARAMS
    }
    return config_map.get(model_name, {})

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Check if required directories exist
        required_dirs = [Config.DATA_DIR, Config.MODELS_DIR]
        for directory in required_dirs:
            if not directory.exists():
                print(f"Warning: Directory {directory} does not exist")
                return False
        
        # Validate thresholds
        if Config.TEXTBLOB_POSITIVE_THRESHOLD <= Config.TEXTBLOB_NEGATIVE_THRESHOLD:
            print("Error: TextBlob positive threshold must be greater than negative threshold")
            return False
        
        if Config.VADER_POSITIVE_THRESHOLD <= Config.VADER_NEGATIVE_THRESHOLD:
            print("Error: VADER positive threshold must be greater than negative threshold")
            return False
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def print_config_summary():
    """Print a summary of current configuration"""
    print("=== CONFIGURATION SUMMARY ===")
    print(f"Project: {Config.PROJECT_NAME} v{Config.VERSION}")
    print(f"Base Directory: {Config.BASE_DIR}")
    print(f"Sample Data Size: {Config.SAMPLE_DATA_SIZE}")
    print(f"Test Size: {Config.TEST_SIZE}")
    print(f"Max Features: {Config.MAX_FEATURES}")
    print(f"Random Seed: {Config.RANDOM_SEED}")
    print(f"Sentiment Labels: {Config.SENTIMENT_LABELS}")
    print("=" * 30)

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Configuration for development environment"""
    DEBUG = True
    SAMPLE_DATA_SIZE = 500
    MAX_FEATURES = 1000

class ProductionConfig(Config):
    """Configuration for production environment"""
    DEBUG = False
    SAMPLE_DATA_SIZE = 10000
    MAX_FEATURES = 10000
    N_JOBS = 4  # Limit for production server

class TestingConfig(Config):
    """Configuration for testing environment"""
    DEBUG = True
    SAMPLE_DATA_SIZE = 100
    MAX_FEATURES = 500
    TEST_SIZE = 0.5

# Configuration factory
def get_config(environment: str = 'development'):
    """Get configuration based on environment"""
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    return config_map.get(environment, DevelopmentConfig)

# Initialize default configuration
config = get_config()

if __name__ == "__main__":
    # Run configuration validation and print summary
    print_config_summary()
    validate_config()
