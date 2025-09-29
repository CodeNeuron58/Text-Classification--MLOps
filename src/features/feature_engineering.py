import os
import yaml
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Custom logger
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import get_logger

logger = get_logger("feature_engineering")


def load_data(train_file, test_file):
    logger.info("Loading processed train and test data...")
    try:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        logger.info(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise


def handle_nan(X_train, X_test):
    logger.info("Handling NaN values in features...")
    try:
        X_train = pd.Series(X_train).fillna("missing_text").values
        X_test = pd.Series(X_test).fillna("missing_text").values
        logger.info("NaN values replaced with 'missing_text'.")
        return X_train, X_test
    except Exception as e:
        logger.error(f"Error handling NaN values: {e}")
        raise


def split_data(train_data, test_data):
    logger.info("Splitting data into features and labels...")
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        logger.info(f"Split completed: X_train {X_train.shape}, y_train {y_train.shape}, "
                    f"X_test {X_test.shape}, y_test {y_test.shape}")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise


def apply_bag_of_words(X_train, X_test, params_file="params.yaml"):
    logger.info(f"Applying tfidf with params from {params_file}")
    try:
        params = yaml.safe_load(open(params_file))["feature_engineering"]["max_features"]
        vectorizer = TfidfVectorizer(max_features=params)
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        logger.info(f"tfidf applied. Train shape: {X_train_vectorized.shape}, "
                    f"Test shape: {X_test_vectorized.shape}")
        return X_train_vectorized, X_test_vectorized
    except Exception as e:
        logger.error(f"Error in tfidf feature extraction: {e}")
        raise


def save_data(train_df, test_df, data_path):
    logger.info(f"Saving feature-engineered data to {data_path}")
    try:
        os.makedirs(data_path, exist_ok=True)
        train_file = os.path.join(data_path, 'train.csv')
        test_file = os.path.join(data_path, 'test.csv')
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        logger.info("Feature data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving feature data: {e}")
        raise


def combining_data(X_train_vectorized, X_test_vectorized, y_train, y_test, data_path):
    logger.info("Combining features with labels...")
    try:
        train_df = pd.DataFrame(X_train_vectorized.toarray())
        train_df["labels"] = y_train
        test_df = pd.DataFrame(X_test_vectorized.toarray())
        test_df["labels"] = y_test
        logger.info(f"Final train shape: {train_df.shape}, test shape: {test_df.shape}")
        save_data(train_df, test_df, data_path)
    except Exception as e:
        logger.error(f"Error combining feature data: {e}")
        raise


def featured_data(train_file, test_file, data_path):
    logger.info("Starting feature engineering pipeline...")
    train_data, test_data = load_data(train_file, test_file)
    X_train, y_train, X_test, y_test = split_data(train_data, test_data)
    X_train, X_test = handle_nan(X_train, X_test)
    X_train_vectorized, X_test_vectorized = apply_bag_of_words(X_train, X_test)
    combining_data(X_train_vectorized, X_test_vectorized, y_train, y_test, data_path)
    logger.info("Feature engineering pipeline completed successfully.")


def main():
    train_file = 'data/interim/train.csv'
    test_file = 'data/interim/test.csv'
    data_path = os.path.join('data', 'processed')
    featured_data(train_file, test_file, data_path)


if __name__ == '__main__':
    main()