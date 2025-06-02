import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import Birch  # Changed from DBSCAN to Birch
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import os  # Added for path joining and directory creation


def build_autoencoder(input_dim, encoding_dim=16):
    """Create feature reconstruction autoencoder"""
    input_layer = Input(shape=(input_dim,))
    # Added a bit more complexity to the autoencoder as in previous suggestions
    encoded = Dense(input_dim // 2, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(input_dim // 2, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def train_unsupervised_models(X_normal, models_save_dir='models'):  # Added models_save_dir parameter
    """Train all unsupervised models on normal data, using BIRCH for clustering."""

    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)
        print(f"Created directory: {models_save_dir}")

    # --- 1. Isolation Forest ---
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(
        contamination='auto',  # Changed to 'auto' or a small float
        random_state=42,
        n_estimators=150  # Increased estimators
    )
    iso_forest.fit(X_normal)
    joblib.dump(iso_forest, os.path.join(models_save_dir, 'isolation_forest.joblib'))
    print("Isolation Forest training complete and model saved.")

    # --- 2. Feature Autoencoder ---
    print("\nTraining Autoencoder...")
    autoencoder = build_autoencoder(X_normal.shape[1])
    # It's good practice to have a validation split and early stopping for NNs
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(
        X_normal, X_normal,
        epochs=100,  # Increased epochs
        batch_size=64,  # Adjusted batch size
        shuffle=True,
        validation_split=0.1,  # Added validation split
        callbacks=[early_stopping],  # Added early stopping
        verbose=1
    )
    autoencoder.save(os.path.join(models_save_dir, 'autoencoder_model.h5'))
    print("Autoencoder training complete and model saved.")

    # --- 3. BIRCH for clustering ---
    print("\nTraining BIRCH...")
    # Key parameters for BIRCH:
    # - threshold: The radius of the subcluster obtained by merging a new sample and the closest subcluster.
    # - branching_factor: The maximum number of CF subclusters in each node.
    # - n_clusters: Number of clusters to return after the final clustering step.
    #   Can be an int, or an instance of another clusterer (e.g., AgglomerativeClustering), or None.
    #   If None, the leaves of the CF Tree are treated as clusters.
    birch_model = Birch(
        threshold=0.5,  # This value is data-dependent and might need tuning.
        branching_factor=50,  # Default is 50.
        n_clusters=None  # Set to None to get leaf CF nodes as clusters, or set an int.
        # Or use another clusterer: e.g., AgglomerativeClustering(n_clusters=10)
    )
    birch_model.fit(X_normal)
    joblib.dump(birch_model, os.path.join(models_save_dir, 'birch_model.joblib'))
    print("BIRCH training complete and model saved.")

    return iso_forest, autoencoder, birch_model
