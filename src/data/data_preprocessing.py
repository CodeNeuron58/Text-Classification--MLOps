import pandas as pd
import numpy as np
import os
import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import get_logger

# Initialize logger
logger = get_logger("data_preprocessing")

# Download required NLTK resources
nltk.download("wordnet")
nltk.download("stopwords")

def load_data(train_file, test_file):
    logger.info("Loading train and test data...")
    try:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error loading train/test data: {e}")
        raise

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized)

def removing_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return " ".join(filtered)

def removing_numbers(text):
    return "".join([ch for ch in text if not ch.isdigit()])

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lower_case(text):
    words = text.split()
    lowered = [word.lower() for word in words]
    return " ".join(lowered)

def removing_urls(text):
    return re.sub(r"http\S+", "", text)

def remove_small_sentences(text):
    words = text.split()
    filtered = [word for word in words if len(word) > 3]
    return " ".join(filtered)

def final_preprocessing(text):
    try:
        text = lemmatization(text)
        text = removing_stopwords(text)
        text = removing_numbers(text)
        text = removing_punctuations(text)
        text = lower_case(text)
        text = removing_urls(text)
        text = remove_small_sentences(text)
        return text
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        raise

def save_data(train_data, test_data, data_path):
    logger.info(f"Saving processed data to {data_path}")
    try:
        os.makedirs(data_path, exist_ok=True)
        train_file = os.path.join(data_path, 'train.csv')
        test_file = os.path.join(data_path, 'test.csv')
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        logger.info(f"Processed train and test data saved successfully at {data_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise

def main():
    train_file = 'data/raw/train.csv'
    test_file = 'data/raw/test.csv'
    data_path = os.path.join('data', 'interim')

    logger.info("Starting preprocessing pipeline...")
    train_data, test_data = load_data(train_file, test_file)

    logger.info("Preprocessing train data...")
    train_data['content'] = train_data['content'].apply(final_preprocessing)
    logger.info("Preprocessing test data...")
    test_data['content'] = test_data['content'].apply(final_preprocessing)

    logger.info("Preprocessing completed. Saving data...")
    save_data(train_data, test_data, data_path)
    logger.info("Preprocessing pipeline completed successfully.")

if __name__ == '__main__':
    main()