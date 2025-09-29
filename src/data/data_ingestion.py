# Import required libraries
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import get_logger

# Initialize logger
logger = get_logger("data_ingestion")

def load_data(data_url):
    logger.info("Loading data from URL")
    try:
        df = pd.read_csv(data_url)
        logger.info("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data from URL: {e}")
        raise

def preprocess(df):
    logger.info("Preprocessing data")
    try:
        df.dropna(inplace=True)
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'] = final_df['sentiment'].map({'happiness': 1, 'sadness': 0})
        logger.info("Preprocessing completed")
        return final_df
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def split_data(df, params_file="params.yaml"):
    logger.info("Splitting data")
    try:
        params = yaml.safe_load(open(params_file))["data_ingestion"]["test_size"]
        train_data, test_data = train_test_split(df, test_size=params, random_state=42)
        logger.info("Data splitting completed")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise

def save_data(train_data, test_data, data_path):
    logger.info("Saving data")
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def run_data_ingestion(data_url, data_path):
    logger.info("Running data ingestion process")
    try:
        df = load_data(data_url)
        df = preprocess(df)
        train_data, test_data = split_data(df)
        save_data(train_data, test_data, data_path)
        logger.info("Data ingestion process completed successfully")
    except Exception as e:
        logger.error(f"Error running data ingestion: {e}")
        raise

def main():
    data_url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
    data_path = os.path.join('data', 'raw')
    run_data_ingestion(data_url, data_path)

if __name__ == '__main__':
    main()