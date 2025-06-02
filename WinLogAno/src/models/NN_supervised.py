# src/models/train_supervised.py
"""
Main script to train a supervised neural network on session-level features.
It loads the session features CSV, trains a classifier, evaluates it,
saves the trained model, and generates visualizations.
"""

import torch
from torch.utils.data import Dataset

import os
import sys
import logging
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score, precision_score, \
    recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure paths and logging
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Create output directories
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models', 'trained')
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, 'models', 'visual')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'nn_training.log'))
    ]
)

# Constants
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "malware_detector_nn_{timestamp}.pth")
SCALER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "feature_scaler_nn_{timestamp}.joblib")
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
TEST_SIZE = 0.25
RANDOM_SEED = 42


class MalwareDetector(nn.Module):
    """Custom neural network architecture"""

    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class MalwareDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data(feature_path: str) -> tuple:
    """Load and preprocess session features"""
    logger.info(f"Loading data from {feature_path}")

    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Data file not found: {feature_path}")

    df = pd.read_csv(feature_path)

    # Ensure label column exists
    if 'label' not in df.columns:
        raise ValueError("Missing required 'label' column in dataset")

    # Identify feature columns
    non_feature_cols = ['session_id', 'session_start_time', 'session_end_time', 'label']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    # Prepare features and labels
    X = df[feature_cols].copy()
    y = df['label'].astype(int).values

    # Handle missing values
    X.fillna(X.median(numeric_only=True), inplace=True)

    # Scale features
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    return X_scaled, y, scaler, feature_cols


def create_data_loaders(X_train, y_train, X_val, y_val):
    """Create PyTorch data loaders with proper tensor conversion"""
    # Convert to numpy arrays if needed
    X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
    X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
    y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)

    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)

    # Handle class imbalance
    class_counts = np.bincount(y_train_np.astype(int))
    class_weights = 1. / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    sample_weights = class_weights[y_train_np.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = MalwareDataset(X_train_tensor, y_train_tensor)
    val_dataset = MalwareDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


def generate_visualizations(model, X_test, y_test, feature_importance_df, visual_dir: str):
    """Generate and save model evaluation visualizations"""
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Get predictions
        with torch.no_grad():
            probs = model(torch.tensor(X_test.values, dtype=torch.float32).to(device))
            probs = probs.cpu().numpy().flatten()

        predictions = (probs >= 0.5).astype(int)

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Malicious'],
                    yticklabels=['Benign', 'Malicious'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(visual_dir, f"nn_confusion_matrix_{timestamp}.png"), bbox_inches='tight')
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(visual_dir, f"nn_roc_curve_{timestamp}.png"), bbox_inches='tight')
        plt.close()

        # Feature Importance (Permutation Importance)
        baseline_score = roc_auc_score(y_test, probs)
        feature_importance = {}

        for col in X_test.columns:
            X_temp = X_test.copy()
            X_temp[col] = np.random.permutation(X_temp[col].values)
            with torch.no_grad():
                permuted_probs = model(
                    torch.tensor(X_temp.values, dtype=torch.float32).to(device)).cpu().numpy().flatten()
            permuted_score = roc_auc_score(y_test, permuted_probs)
            feature_importance[col] = baseline_score - permuted_score

        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        features, importance = zip(*sorted_importance[:20])
        plt.figure(figsize=(10, 8))
        ax = sns.barplot(x=importance, y=features, palette="viridis")
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature Name')
        for p in ax.patches:
            width = p.get_width()
            plt.text(width * 1.01, p.get_y() + p.get_height() / 2.,
                     f'{width:.4f}', ha='left', va='center')
        plt.savefig(os.path.join(visual_dir, f"nn_feature_importance_{timestamp}.png"), bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"Error generating visualizations: {e}", exc_info=True)


def train_evaluate_model(features_df: pd.DataFrame, target_column: str = 'label'):
    """Train and evaluate neural network model"""
    # Validate input
    if features_df.empty:
        logger.error("Input DataFrame is empty. Aborting training.")
        return None, None

    # Identify feature columns
    non_feature_cols = ['session_id', 'session_start_time', 'session_end_time', target_column]
    feature_cols = [col for col in features_df.columns if col not in non_feature_cols]

    if not feature_cols:
        logger.error("No feature columns identified. Check your data.")
        return None, None

    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Prepare data
    X = features_df[feature_cols]
    y = features_df[target_column].astype(int)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_test, y_test)

    # Initialize model
    model = MalwareDetector(X_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training setup
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Training loop
    best_loss = float('inf')
    best_epoch = 0
    early_stop_counter = 0

    train_losses, val_losses = [], []

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0

        # Inside train_model():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # shape: (batch_size,)
            loss = criterion(outputs, labels)  # labels should also be (batch_size,)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).flatten()
                running_loss += criterion(outputs, labels).item() * inputs.size(0)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_auc = roc_auc_score(all_labels, all_preds)

        logger.info(f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(),
                       MODEL_SAVE_PATH.replace("{timestamp}", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step(val_loss)

    # Final Evaluation
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
        probs = model(test_tensor).cpu().numpy().flatten()

    predictions = (probs >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1': f1_score(y_test, predictions, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probs)
    }

    logger.info("\n=== Neural Network Evaluation ===")
    for name, value in metrics.items():
        logger.info(f"{name.capitalize()}: {value:.4f}")

    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, predictions, zero_division=0))

    # Feature importance analysis
    feature_importance = {}
    for col in X.columns:
        X_temp = X_test.copy()
        X_temp[col] = np.random.permutation(X_temp[col].values)
        with torch.no_grad():
            permuted_probs = model(torch.tensor(X_temp.values, dtype=torch.float32).to(device)).cpu().numpy().flatten()
        feature_importance[col] = roc_auc_score(y_test, probs) - roc_auc_score(y_test, permuted_probs)

    fi_df = pd.DataFrame(feature_importance.items(), columns=['feature', 'importance'])
    fi_df = fi_df.sort_values('importance', ascending=False)

    logger.info("\n--- Top 10 Feature Importances ---")
    logger.info(fi_df.head(10).to_string(index=False))

    # Generate visualizations
    generate_visualizations(model, X_test, y_test, fi_df, VISUALIZATION_DIR)

    return model, metrics


def main():
    """Main training pipeline"""
    # Path to session features
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'session_features.csv')

    logger.info(f"Starting neural network training with data: {data_path}")

    try:
        # Load and validate data
        df = pd.read_csv(data_path)

        if df.empty:
            logger.error("Loaded DataFrame is empty. Aborting training.")
            sys.exit(1)

        if 'label' not in df.columns:
            logger.error("Missing 'label' column in data.")
            sys.exit(1)

        label_counts = df['label'].value_counts()
        if len(label_counts) < 2:
            logger.error(f"Insufficient classes in target: {label_counts.to_dict()}")
            sys.exit(1)

        # Train model
        model, metrics = train_evaluate_model(df)

        if model:
            logger.info("Neural network training completed successfully")
        else:
            logger.error("Neural network training failed")
            sys.exit(1)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Create logs directory if needed
    os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)
    main()