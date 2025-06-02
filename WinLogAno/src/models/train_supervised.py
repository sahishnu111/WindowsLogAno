# src/models/train_model.py
"""
Main script to train a Random Forest model on session-level features.
Loads features, trains classifier, evaluates, saves model, and generates visualizations.
"""
import pandas as pd
import numpy as np
import os
import sys
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, auc
)

# Configure paths and logging
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Create output directories
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models', 'trained')
VISUAL_DIR = os.path.join(PROJECT_ROOT, 'models', 'visual')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'model_training.log'))
    ]
)


def load_data(feature_path: str) -> pd.DataFrame:
    """Loads the session features CSV file."""
    if not os.path.exists(feature_path):
        logger.error(f"Feature file not found: {feature_path}")
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    try:
        df = pd.read_csv(feature_path)
        logger.info(f"Loaded features from {feature_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handles infinities and NaNs in the data."""
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.isnull().any().any():
        null_count = df.isnull().sum().sum()
        logger.warning(f"NaN values found in {null_count} cells. Filling with 0.")
        df.fillna(0, inplace=True)
    return df


def generate_visualizations(model, X_test, y_test, feature_importance_df, visual_dir: str):
    """Generates and saves evaluation visualizations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Malware'],
                    yticklabels=['Normal', 'Malware'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(visual_dir, f"confusion_matrix_{timestamp}.png")
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix to {cm_path}")

        # 2. ROC Curve
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            roc_path = os.path.join(visual_dir, f"roc_curve_{timestamp}.png")
            plt.savefig(roc_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved ROC curve to {roc_path}")

        # 3. Feature Importance Plot
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        ax = sns.barplot(x='importance', y='feature', data=top_features, palette="viridis")
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        for p in ax.patches:
            width = p.get_width()
            plt.text(width * 1.01, p.get_y() + p.get_height() / 2.,
                     f'{width:.4f}', ha='left', va='center')
        fi_path = os.path.join(visual_dir, f"feature_importance_{timestamp}.png")
        plt.savefig(fi_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved feature importance plot to {fi_path}")

    except Exception as e:
        logger.error(f"Visualization error: {e}", exc_info=True)


def train_evaluate_model(features_df: pd.DataFrame, target_column: str = 'label'):
    """Trains and evaluates a Random Forest classifier."""
    # Validate input data
    if features_df.empty:
        logger.error("Input DataFrame is empty. Aborting training.")
        return None

    if target_column not in features_df.columns:
        logger.error(f"Target column '{target_column}' missing")
        return None

    # Ensure target is binary and clean
    if target_column in features_df.columns:
        # Convert to binary (if needed)
        features_df[target_column] = features_df[target_column].astype(int)

        # Check for only 0/1 values
        if not all(features_df[target_column].isin([0, 1])):
            logger.warning("Label column contains non-binary values. Converting to binary...")
            features_df[target_column] = (features_df[target_column] > 0).astype(int)

    # Identify feature columns
    non_feature_cols = ['session_id', 'session_start_time', 'session_end_time', target_column]
    feature_cols = [col for col in features_df.columns if col not in non_feature_cols]

    if not feature_cols:
        logger.error("No features identified")
        return None

    logger.info(f"Using {len(feature_cols)} features for training")

    # Prepare data
    X = features_df[feature_cols]
    y = features_df[target_column].astype(int)
    X = preprocess_data(X.copy())

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    logger.info("Training Random Forest model...")

    try:
        model.fit(X_train, y_train)
        logger.info("Model training completed")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return None

    # Evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
    }

    logger.info("\n=== Model Evaluation ===")
    for name, value in metrics.items():
        logger.info(f"{name.capitalize()}: {value:.4f}")

    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, zero_division=0))

    # Feature importances
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)

    logger.info("\nTop 10 Features:")
    logger.info(fi_df.head(10).to_string(index=False))

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"malware_detector_{timestamp}.joblib"
    model_path = os.path.join(MODEL_SAVE_DIR, model_name)

    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Model save failed: {e}", exc_info=True)

    # Generate visualizations
    generate_visualizations(model, X_test, y_test, fi_df, VISUAL_DIR)

    return model, metrics


if __name__ == "__main__":
    # Path to session features
    FEATURE_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'session_features.csv')
    logger.info(f"Starting training pipeline with features: {FEATURE_PATH}")

    try:
        # Load and validate data
        df = load_data(FEATURE_PATH)

        if df.empty:
            logger.error("Loaded empty DataFrame. Aborting.")
            sys.exit(1)

        if 'label' not in df.columns:
            logger.error("Missing session_label column")
            sys.exit(1)

        label_counts = df['label'].value_counts()
        logger.info(f"Class distribution:\n{label_counts}")

        if len(label_counts) < 2:
            logger.error("Insufficient classes for classification")
            sys.exit(1)

        # Train and evaluate
        model, metrics = train_evaluate_model(df)
        if model:
            logger.info("Training completed successfully")
        else:
            logger.error("Training failed")
            sys.exit(1)

    except FileNotFoundError:
        logger.error("Feature file not found. Run build_features.py first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        sys.exit(1)