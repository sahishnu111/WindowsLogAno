# src/on_device/prototype.py

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.models import load_model as keras_load_model
from src.data_processing.log_parser import preprocess_logs
from src.features.build_features import markov_features, fft_features, sequence_features, keyword_features


class HybridModelPyTorch(nn.Module):
    """PyTorch implementation of the hybrid anomaly detection model"""

    def __init__(self, agg_dim, seq_length, vocab_size, unsup_dim,
                 embedding_dim=32, lstm_units=64, dense_units=64):
        super(HybridModelPyTorch, self).__init__()

        # Aggregate feature branch
        self.agg_dense = nn.Sequential(
            nn.Linear(agg_dim, dense_units),
            nn.ReLU(),
            nn.BatchNorm1d(dense_units)
        )

        # Sequence branch
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True)

        # Unsupervised features branch
        self.unsup_dense = nn.Sequential(
            nn.Linear(unsup_dim, dense_units // 2),
            nn.ReLU()
        )

        # Combined processing
        combined_dim = dense_units + lstm_units + (dense_units // 2)
        self.combined = nn.Sequential(
            nn.Linear(combined_dim, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, 1),
            nn.Sigmoid()
        )

    def forward(self, x_agg, x_seq, x_unsup):
        # Process each branch
        agg_out = self.agg_dense(x_agg)

        embedded = self.embedding(x_seq)
        _, (hidden, _) = self.lstm(embedded)
        seq_out = hidden[-1]  # Last layer hidden state

        unsup_out = self.unsup_dense(x_unsup)

        # Combine features
        combined = torch.cat((agg_out, seq_out, unsup_out), dim=1)
        return self.combined(combined)


class RealTimeDetector:
    """Real-time log anomaly detection system with adaptive thresholding"""

    def __init__(self, model_dir, baseline_stats=None, device='cpu'):
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.baseline = baseline_stats or {}
        self.cluster_history = set()

        # Load models
        self.load_models()

        # Initialize thresholds
        self.thresholds = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.6
        }

        print(f"RealTimeDetector initialized on {device} with:")
        print(f"- Hybrid model: vocab_size={self.vocab_size}, seq_length={self.seq_length}")
        print(f"- Active thresholds: Critical>{self.thresholds['critical']}, High>{self.thresholds['high']}")

    def load_models(self):
        """Load all required models and resources"""
        # Load PyTorch model
        model_path = os.path.join(self.model_dir, 'hybrid_anomaly_classifier_pytorch_best.pt')
        self.model = self.load_pytorch_model(model_path)

        # Load tokenizer and scaler
        self.tokenizer = joblib.load(os.path.join(self.model_dir, 'hybrid_tokenizer.joblib'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'feature_scaler.joblib'))

        # Load unsupervised models
        self.iso_forest = joblib.load(os.path.join(self.model_dir, 'isolation_forest.joblib'))
        self.autoencoder = keras_load_model(os.path.join(self.model_dir, 'autoencoder_model.h5'))

        # Get model parameters
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.seq_length = joblib.load(os.path.join(self.model_dir, 'sequence_length.joblib'))

        # Baseline initialization
        if not self.baseline:
            self.baseline = {
                'rec_error': [],
                'seq_error': []
            }

    def load_pytorch_model(self, model_path):
        """Load PyTorch model with architecture parameters"""
        # First load the state dict to get parameters
        checkpoint = torch.load(model_path, map_location=self.device)

        # Reconstruct model architecture
        model = HybridModelPyTorch(
            agg_dim=checkpoint['agg_dim'],
            seq_length=checkpoint['seq_length'],
            vocab_size=checkpoint['vocab_size'],
            unsup_dim=checkpoint['unsup_dim']
        ).to(self.device)

        # Load weights
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    def process_log_window(self, log_df):
        """Process a window of logs for anomaly detection"""
        # Preprocessing
        cleaned_df = preprocess_logs(log_df)

        # Feature extraction
        features = self.extract_features(cleaned_df)

        # Prepare model inputs
        inputs = self.prepare_model_inputs(features)

        # Model prediction
        with torch.no_grad():
            prediction = self.model(*inputs)
            proba = prediction.item()

        # Get unsupervised features
        unsup_features = self.get_unsupervised_features(features['agg_features'])

        # Determine alert level
        alert_level = self.determine_alert_level(
            proba,
            features,
            unsup_features
        )

        return alert_level, proba, features

    def extract_features(self, cleaned_df):
        """Extract all required features from log data"""
        features = {}

        # Markov transition features
        markov = markov_features(cleaned_df)
        features.update(markov)

        # FFT features
        time_series = cleaned_df.groupby(pd.Grouper(key='timestamp', freq='1min')).size()
        features['beaconing_score'] = fft_features(time_series.fillna(0).values)

        # Sequence features
        sequences = cleaned_df.groupby('session_id')['event_id'].apply(list)
        features['predictability'], features['pair_entropy'], _ = sequence_features(sequences)

        # Keyword features
        features['malicious_tool_freq'] = keyword_features(cleaned_df['message'].tolist())

        # Aggregate features for model
        agg_feature_names = [
            'error_persistence', 'warn_to_error', 'info_stability',
            'beaconing_score', 'predictability', 'pair_entropy',
            'malicious_tool_freq'
        ]
        features['agg_features'] = np.array([[features[k] for k in agg_feature_names]])

        return features

    def prepare_model_inputs(self, features):
        """Prepare features for model input"""
        # Scale aggregate features
        agg_scaled = self.scaler.transform(features['agg_features'])

        # Prepare sequence data
        event_ids = features.get('sequence', [])
        if not event_ids:  # Handle empty sequences
            event_ids = [0]

        # Tokenize and pad sequence
        seq_tokenized = self.tokenizer.texts_to_sequences([list(map(str, event_ids))])
        seq_padded = np.zeros((1, self.seq_length))
        seq_padded[0, :len(seq_tokenized[0])] = seq_tokenized[0][:self.seq_length]

        # Unsupervised features
        unsup_features = self.get_unsupervised_features(agg_scaled)

        # Convert to tensors
        agg_tensor = torch.tensor(agg_scaled, dtype=torch.float32).to(self.device)
        seq_tensor = torch.tensor(seq_padded, dtype=torch.long).to(self.device)
        unsup_tensor = torch.tensor(unsup_features, dtype=torch.float32).to(self.device)

        return agg_tensor, seq_tensor, unsup_tensor

    def get_unsupervised_features(self, agg_features):
        """Generate unsupervised features for a single window"""
        # Reconstruction error
        recon = self.autoencoder.predict(agg_features)
        rec_error = np.mean(np.square(agg_features - recon))

        # Isolation forest score
        iso_score = self.iso_forest.decision_function(agg_features)[0]

        # Normalize to [0, 1]
        rec_error_norm = (rec_error - self.baseline['rec_error_min']) / \
                         (self.baseline['rec_error_max'] - self.baseline['rec_error_min'] + 1e-9)
        iso_score_norm = (iso_score - self.baseline['iso_score_min']) / \
                         (self.baseline['iso_score_max'] - self.baseline['iso_score_min'] + 1e-9)

        return np.array([[rec_error_norm, iso_score_norm]])

    def determine_alert_level(self, proba, features, unsup_features):
        """Determine alert level based on probability and features"""
        # Critical alerts (C2 beaconing + tools)
        if (features['beaconing_score'] > 15 and
                features['malicious_tool_freq'] > 10 and
                proba > self.thresholds['critical']):
            return "CRITICAL"

        # High confidence (anomalous sequences)
        if (features['predictability'] < 0.4 and
                features['pair_entropy'] > 2.5 and
                proba > self.thresholds['high']):
            return "HIGH"

        # Novel pattern detection
        if (unsup_features[0, 0] > 0.8 or  # High reconstruction error
            unsup_features[0, 1] < -0.5) and  # Low isolation score
            proba > self.thresholds['medium']:
            return "MEDIUM"

        return "NORMAL"

    def update_baseline(self, new_normal_data):
        """Update baseline statistics with new normal data"""
        # Update reconstruction baseline
        new_rec_errors = []
        new_iso_scores = []

        for data in new_normal_data:
            features = self.extract_features(data)
            agg_features = features['agg_features']

            # Calculate reconstruction error
            recon = self.autoencoder.predict(agg_features)
            rec_error = np.mean(np.square(agg_features - recon))
            new_rec_errors.append(rec_error)

            # Isolation forest score
            iso_score = self.iso_forest.decision_function(agg_features)[0]
            new_iso_scores.append(iso_score)

        # Update min/max ranges
        self.baseline['rec_error_min'] = min(self.baseline.get('rec_error_min', float('inf')), min(new_rec_errors))
        self.baseline['rec_error_max'] = max(self.baseline.get('rec_error_max', 0), max(new_rec_errors))
        self.baseline['iso_score_min'] = min(self.baseline.get('iso_score_min', float('inf')), min(new_iso_scores))
        self.baseline['iso_score_max'] = max(self.baseline.get('iso_score_max', 0), max(new_iso_scores))

        # Update cluster history
        cluster_labels = self.iso_forest.fit_predict(np.vstack([d['agg_features'] for d in new_normal_data))
        self.cluster_history.update(set(cluster_labels))

        # Helper function for windowed processing


def create_log_windows(log_df, window_size='5T'):
    """Create time-based windows from log data"""
    log_df = log_df.sort_values('timestamp')
    windows = []

    # Generate time-based windows
    for _, window_df in log_df.groupby(pd.Grouper(key='timestamp', freq=window_size)):
        if not window_df.empty:
            windows.append(window_df)

    return windows


def run_real_time_detection(log_source, model_dir, output_handler):
    """Main function for real-time detection"""
    # Initialize detector
    detector = RealTimeDetector(model_dir, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Process log source (could be file, stream, etc.)
    for raw_logs in log_source:
        # Preprocess logs
        cleaned_logs = preprocess_logs(raw_logs)

        # Create time windows
        windows = create_log_windows(cleaned_logs)

        for window in windows:
            alert_level, proba, features = detector.process_log_window(window)

            if alert_level != "NORMAL":
                output_handler({
                    'timestamp': window['timestamp'].iloc[0],
                    'alert_level': alert_level,
                    'probability': proba,
                    'features': features,
                    'sample_events': window[['timestamp', 'event_id', 'message']].head(2).to_dict('records')
                })

                # Optional: Update baseline with normal data periodically
                # if time_to_update_baseline():
                #     detector.update_baseline(recent_normal_data)