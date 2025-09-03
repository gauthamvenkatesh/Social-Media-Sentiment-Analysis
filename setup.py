#!/usr/bin/env python3
"""
Setup script for Social Media Sentiment Analysis Project
This script helps set up the project environment and dependencies
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              Social Media Sentiment Analysis                 ║
    ║                     Setup Script v1.0                       ║
    ║                                                              ║
    ║  This script will help you set up the complete project      ║
    ║  environment for sentiment analysis.                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version {sys.version.split()[0]} is compatible")
    return True

def create_project_structure():
    """Create the project directory structure"""
    print("\n📁 Creating project directory structure...")
    
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "data/results",
        "models",
        "notebooks",
        "src",
        "tests",
        "docs",
        "templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}/")
    
    print("✅ Project structure created successfully!")

def install_requirements():
    """Install required Python packages"""
    print("\n📦 Installing required packages...")
    
    packages = [
        "pandas==1.5.3",
        "numpy==1.24.3",
        "matplotlib==3.7.1",
        "seaborn==0.12.2",
        "plotly==5.14.1",
        "scikit-learn==1.2.2",
        "nltk==3.8.1",
        "textblob==0.17.1",
        "vaderSentiment==3.3.2",
        "wordcloud==1.9.2",
        "streamlit==1.23.1",
        "requests==2.31.0",
        "tweepy==4.14.0",
        "jupyter",
        "notebook",
        "ipywidgets"
    ]
    
    try:
        for package in packages:
            print(f"   Installing {package.split('==')[0]}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
        
        print("✅ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    import nltk
    
    nltk_downloads = [
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4',
        'vader_lexicon',
        'averaged_perceptron_tagger'
    ]
    
    try:
        for item in nltk_downloads:
            print(f"   Downloading {item}...")
            nltk.download(item, quiet=True)
        
        print("✅ NLTK data downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def create_sample_files():
    """Create sample configuration and data files"""
    print("\n📝 Creating sample files...")
    
    # Create sample .env file
    env_content = """# Environment variables for Social Media Sentiment Analysis
# Copy this file to .env and fill in your API credentials

# Twitter API Credentials
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
TWITTER_ACCESS_TOKEN=your_twitter_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret_here

# Reddit API Credentials
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=SentimentAnalysis/1.0

# Other Configuration
ENVIRONMENT=development
DEBUG=True
"""
    
    with open('.env.sample', 'w') as f:
        f.write(env_content)
    
    # Create gitignore file
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data files
*.csv
*.json
*.pkl
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/*.pkl
models/*.joblib

# Logs
*.log

# Environment variables
.env

# Jupyter Notebooks
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    # Create placeholder files
    placeholder_files = [
        'data/raw/.gitkeep',
        'data/processed/.gitkeep',
        'data/results/.gitkeep',
        'models/.gitkeep',
        'tests/.gitkeep',
        'docs/.gitkeep'
    ]
    
    for file_path in placeholder_files:
        Path(file_path).touch()
    
    print("✅ Sample files created successfully!")

def run_initial_test():
    """Run a quick test to verify setup"""
    print("\n🧪 Running initial setup test...")
    
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import nltk
        from textblob import TextBlob
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from sklearn.linear_model import LogisticRegression
        
        # Test basic functionality
        analyzer = SentimentIntensityAnalyzer()
        test_text = "I love this project!"
        
        # TextBlob test
        blob = TextBlob(test_text)
        tb_sentiment = blob.sentiment.polarity
        
        # VADER test
        vader_scores = analyzer.polarity_scores(test_text)
        
        print(f"   Test text: '{test_text}'")
        print(f"   TextBlob polarity: {tb_sentiment:.3f}")
        print(f"   VADER compound: {vader_scores['compound']:.3f}")
        
        print("✅ All tests passed! Setup is complete!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def display_next_steps():
    """Display next steps for the user"""
    next_steps = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                        Next Steps                            ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  1. 🚀 Run the main analysis:                               ║
    ║     python sentiment_analysis.py                            ║
    ║                                                              ║
    ║  2. 🌐 Launch the web dashboard:                            ║
    ║     streamlit run streamlit_app.py                          ║
    ║                                                              ║
    ║  3. 📓 Explore with Jupyter:                                ║
    ║     jupyter notebook notebooks/                             ║
    ║                                                              ║
    ║  4. 📊 Check the data exploration notebook:                 ║
    ║     notebooks/data_exploration.ipynb                        ║
    ║                                                              ║
    ║  5. ⚙️  Customize settings in config.py                     ║
    ║                                                              ║
    ║  6. 📚 Add your own data to data/raw/                       ║
    ║                                                              ║
    ║  7. 🔑 Configure API keys in .env file (optional)          ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                     Useful Commands                          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  • View project structure: tree . (or ls -la)              ║
    ║  • Test configuration: python config.py                     ║
    ║  • Update packages: pip install -r requirements.txt --upgrade║
    ║  • Run tests: python -m pytest tests/                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(next_steps)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Creating project structure", create_project_structure),
        ("Installing requirements", install_requirements),
        ("Downloading NLTK data", download_nltk_data),
        ("Creating sample files", create_sample_files),
        ("Running initial test", run_initial_test)
    ]
    
    for step_name, step_func in steps:
        try:
            if not step_func():
                print(f"❌ Failed at step: {step_name}")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n\n⚠️  Setup interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error during {step_name}: {e}")
            sys.exit(1)
    
    # Display success message and next steps
    print("\n" + "="*70)
    print("🎉 SETUP COMPLETED SUCCESSFULLY! 🎉")
    print("="*70)
    
    display_next_steps()

if __name__ == "__main__":
    main()
