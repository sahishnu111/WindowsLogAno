# src/on_device/prototype.py

import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from datetime import datetime, timedelta

try:
    from tensorflow.keras.models import load_model as keras_load_model
except ImportError:
    print("Warning: TensorFlow/Keras not installed. Keras autoencoder will not be available.")
    keras_load_model = None

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import custom modules
try:
    from src.data_processing.log_parser import parse_log_files, clean_log_data
    from src.features.build_features import (
        calculate_markov_features,
        calculate_fft_features,
        calculate_sequence_features,
        calculate_keyword_features
    )
except ImportError as e:
    print(f"Error importing from src: {e}")
    print("Ensure you are running from the project root and all __init__.py files are present.")
    sys.exit(1)


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
        agg_out = self.agg_dense(x_agg)
        embedded = self.embedding(x_seq)
        _, (hidden, _) = self.lstm(embedded)
        seq_out = hidden[-1]  # Last layer hidden state
        unsup_out = self.unsup_dense(x_unsup)
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
        print(f"RealTimeDetector initialized on {device}")

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
        if keras_load_model:
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
        # Load the state dict to get parameters
        checkpoint = torch.load(model_path, map_location=self.device)

        # Reconstruct model
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

    def preprocess_window(self, raw_window_df):
        """Preprocess a window of raw logs"""
        # Placeholder - replace with your actual preprocessing logic
        # This should include:
        #   - Timestamp parsing
        #   - Event type classification
        #   - Session identification
        #   - Hex code extraction
        cleaned_df = raw_window_df.copy()
        if 'timecreated' in cleaned_df.columns:
            cleaned_df['timecreated'] = pd.to_datetime(cleaned_df['timecreated'], errors='coerce', utc=True)
            cleaned_df.dropna(subset=['timecreated'], inplace=True)
        return cleaned_df

    def extract_features(self, cleaned_df):
        """Extract all required features from log data"""
        features = {}

        # Markov transition features
        if not cleaned_df.empty:
            markov = calculate_markov_features(cleaned_df)
            features.update(markov)

        # FFT features
        if not cleaned_df.empty and 'timecreated' in cleaned_df.columns:
            time_series = cleaned_df.set_index('timecreated').resample('1min').size()
            features['beaconing_score'] = calculate_fft_features(time_series.fillna(0).values)
        else:
            features['beaconing_score'] = 0.0

        # Sequence features
        if not cleaned_df.empty:
            predictability, entropy, max_chain = calculate_sequence_features(cleaned_df)
            features['predictability'] = predictability
            features['pair_entropy'] = entropy
        else:
            features['predictability'] = 1.0
            features['pair_entropy'] = 0.0

        # Keyword features
        if not cleaned_df.empty and 'message' in cleaned_df.columns:
            messages = cleaned_df['message'].tolist()
            features['malicious_tool_freq'] = calculate_keyword_features(messages)
        else:
            features['malicious_tool_freq'] = 0.0

        # Aggregate features for model
        agg_feature_names = [
            'error_persistence', 'warn_to_error', 'info_stability',
            'beaconing_score', 'predictability', 'pair_entropy',
            'malicious_tool_freq'
        ]
        features['agg_features'] = np.array([[features[k] for k in agg_feature_names]])

        # Store event IDs for sequence processing
        features['event_ids'] = cleaned_df['id'].tolist() if 'id' in cleaned_df.columns else []

        return features

    def prepare_model_inputs(self, features):
        """Prepare features for model input"""
        # Scale aggregate features
        agg_scaled = self.scaler.transform(features['agg_features'])

        # Prepare sequence data
        event_ids = features.get('event_ids', [])
        if not event_ids:  # Handle empty sequences
            seq_tokenized = [[0]]
        else:
            seq_tokenized = self.tokenizer.texts_to_sequences([list(map(str, event_ids))])

        # Pad sequence
        from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences
        seq_padded = keras_pad_sequences(seq_tokenized, maxlen=self.seq_length,
                                         padding='post', truncating='post', value=0)

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
        if self.autoencoder:
            recon = self.autoencoder.predict(agg_features)
            rec_error = np.mean(np.square(agg_features - recon))
        else:
            rec_error = 0.0

        # Isolation forest score
        iso_score = self.iso_forest.decision_function(agg_features)[0]

        # Normalize to [0, 1] using baseline
        rec_error_norm = (rec_error - self.baseline.get('rec_error_min', 0)) / \
                         (self.baseline.get('rec_error_max', 1) - self.baseline.get('rec_error_min', 0) + 1e-9)
        iso_score_norm = (iso_score - self.baseline.get('iso_score_min', -0.5)) / \
                         (self.baseline.get('iso_score_max', 0.5) - self.baseline.get('iso_score_min', -0.5) + 1e-9)

        # Clip to avoid extreme values
        rec_error_norm = np.clip(rec_error_norm, 0, 1)
        iso_score_norm = np.clip(iso_score_norm, 0, 1)

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
        if ((unsup_features[0, 0] > 0.8 or  # High reconstruction error
             unsup_features[0, 1] < -0.5) and  # Low isolation score
                proba > self.thresholds['medium']):
            return "MEDIUM"

        return "NORMAL"

    def process_log_window(self, raw_window_df):
        """Process a window of logs for anomaly detection"""
        # Preprocessing
        cleaned_df = self.preprocess_window(raw_window_df)
        if cleaned_df.empty:
            return "NORMAL", 0.0, {}

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
        alert_level = self.determine_alert_level(proba, features, unsup_features)

        return alert_level, proba, features


def create_log_windows(log_df, window_size='5T'):
    """Create time-based windows from log data"""
    if log_df.empty or 'timecreated' not in log_df.columns:
        return []

    # Ensure timestamp column is datetime
    log_df['timecreated'] = pd.to_datetime(log_df['timecreated'], errors='coerce', utc=True)
    log_df = log_df.dropna(subset=['timecreated']).sort_values('timecreated')

    windows = []
    for _, group in log_df.groupby(pd.Grouper(key='timecreated', freq=window_size)):
        if not group.empty:
            windows.append(group)
    return windows


def run_real_time_detection(log_file_path, model_dir, output_handler, window_size='5T'):
    """Main function for real-time detection from a log file"""
    # Load log data
    print(f"Loading log data from {log_file_path}...")
    log_df = pd.read_csv(log_file_path)

    # Create time windows
    windows = create_log_windows(log_df, window_size=window_size)
    print(f"Created {len(windows)} time windows for processing")

    # Initialize detector
    detector = RealTimeDetector(model_dir, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Process each window
    for i, window in enumerate(windows):
        start_time = time.time()
        alert_level, proba, features = detector.process_log_window(window)
        proc_time = time.time() - start_time

        window_start = window['timecreated'].iloc[0]
        output_handler({
            'window_id': i,
            'timestamp': window_start,
            'alert_level': alert_level,
            'probability': proba,
            'processing_time': proc_time,
            'features': features,
            'num_events': len(window)
        })

        print(f"Window {i} ({window_start}) - {alert_level} alert - {proba:.2%} - {proc_time:.2f}s")


def simple_output_handler(result):
    """Simple output handler for detection results"""
    timestamp = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Window {result['window_id']} ({result['num_events']} events)")
    print(f"  Alert: {result['alert_level']}, Probability: {result['probability']:.2%}")
    print(f"  Processing time: {result['processing_time']:.2f}s")

    if result['alert_level'] != "NORMAL":
        print("  Features:")
        print(f"    - Beaconing score: {result['features']['beaconing_score']:.2f}")
        print(f"    - Malicious tool freq: {result['features']['malicious_tool_freq']:.2f}")
        print(f"    - Predictability: {result['features']['predictability']:.2f}")
        print(f"    - Pair entropy: {result['features']['pair_entropy']:.2f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Real-time Log Anomaly Detector')
    parser.add_argument('log_file', help='Path to CSV log file')
    parser.add_argument('--model_dir', default='models', help='Path to model directory')
    parser.add_argument('--window_size', default='5T', help='Time window size (e.g., 5T for 5 minutes)')

    args = parser.parse_args()

    # Run detection
    run_real_time_detection(
        log_file_path=args.log_file,
        model_dir=args.model_dir,
        output_handler=simple_output_handler,
        window_size=args.window_size
    )