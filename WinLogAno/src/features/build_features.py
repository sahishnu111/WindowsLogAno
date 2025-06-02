# src/features/build_features.py
"""
Main script to generate a session-level feature dataset for model training.
It uses MalwareFeatureExtractor (which uses AdvancedFeatureEngineer)
to calculate features for each session from cleaned event logs.
"""
from collections import Counter

import pandas as pd
import numpy as np
import os
import sys
from typing import Set, Tuple, Any, List, Dict, Optional
import logging

from IPython.core.display_functions import display

# Ensure src is in path for direct execution, though usually run from project root
SCRIPT_DIR_BF = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_BF = os.path.abspath(os.path.join(SCRIPT_DIR_BF, '..', '..')) # Should be WinLogAno
if PROJECT_ROOT_BF not in sys.path:
   sys.path.insert(0, PROJECT_ROOT_BF)

from features.advanced_features import MalwareFeatureExtractor # Correct relative import

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_session_level_features(cleaned_event_df: pd.DataFrame,
                                    top_pairs_for_predictability: Optional[Set[Tuple[Any, Any]]] = None,
                                    hex_whitelist_set: Optional[Set[str]] = None) -> pd.DataFrame:
    """
    Generates a DataFrame of features aggregated at the session level.

    Args:
        cleaned_event_df: DataFrame of cleaned event logs. Must include 'session_id',
                          'timestamp', 'event_id', 'level', 'message', 'source', and 'label'.
                          Should also include per-event lexical features like
                          'malicious_keywords_count' and 'hex_codes_count' from data_processing.
        top_pairs_for_predictability: Optional pre-calculated set of top event pairs from normal data.
        hex_whitelist_set: Optional set of whitelisted hex codes.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a session and columns are
                      the engineered features and the session label.
    """
    if cleaned_event_df.empty or 'session_id' not in cleaned_event_df.columns:
        logger.error("Input DataFrame is empty or missing 'session_id'. Cannot generate session features.")
        return pd.DataFrame()

    required_cols = ['timestamp', 'event_id', 'level', 'message', 'source', 'label',
                     'malicious_keywords_count', 'hex_codes_count'] # From data_processing
    missing_cols = [col for col in required_cols if col not in cleaned_event_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in cleaned_event_df: {missing_cols}")
        return pd.DataFrame()

    logger.info(f"Starting feature generation for {cleaned_event_df['session_id'].nunique()} sessions.")

    # Initialize the feature extractor
    # Pass top_pairs and hex_whitelist if they are pre-calculated from a training/normal dataset
    # For now, MalwareFeatureExtractor will use its defaults if these are None.
    feature_extractor = MalwareFeatureExtractor(
        top_pairs=top_pairs_for_predictability,
        hex_whitelist=hex_whitelist_set
    )

    all_session_feature_records = []

    for session_id, session_logs_df in cleaned_event_df.groupby('session_id'):
        if session_logs_df.empty:
            continue

        logger.debug(f"Processing session_id: {session_id} with {len(session_logs_df)} events.")

        # Extract advanced features using MalwareFeatureExtractor
        # This method now returns a flat dictionary of all features.
        try:
            extracted_features = feature_extractor.extract_all_features(session_logs_df)

            # Basic session aggregates not covered by MalwareFeatureExtractor explicitly,
            # but some are calculated internally (like duration).
            # We can add more here if needed, or ensure MalwareFeatureExtractor is comprehensive.

            # Example: Aggregating pre-calculated per-event lexical features
            extracted_features['session_total_malicious_keywords'] = session_logs_df['malicious_keywords_count'].sum()
            extracted_features['session_avg_malicious_keywords_per_event'] = session_logs_df['malicious_keywords_count'].mean()
            extracted_features['session_total_hex_codes'] = session_logs_df['hex_codes_count'].sum()
            extracted_features['session_avg_hex_codes_per_event'] = session_logs_df['hex_codes_count'].mean()

            # Event counts
            extracted_features['session_event_count'] = len(session_logs_df)
            extracted_features['session_unique_event_ids_count'] = session_logs_df['event_id'].nunique()

            # Session label (max of event labels in the session)
            extracted_features['session_label'] = session_logs_df['label'].max()
            extracted_features['session_id'] = session_id # Ensure session_id is part of the record
            extracted_features['session_start_time'] = session_logs_df['timestamp'].min()
            extracted_features['session_end_time'] = session_logs_df['timestamp'].max()

            all_session_feature_records.append(extracted_features)
        except Exception as e:
            logger.error(f"Error processing session {session_id}: {e}", exc_info=True)
            continue # Skip this session if feature extraction fails

    if not all_session_feature_records:
        logger.warning("No session features were generated.")
        return pd.DataFrame()

    session_features_df = pd.DataFrame(all_session_feature_records)

    # Define the feature list for the model based on EDA1
    # This also helps in ordering and selecting columns for the final features.csv
    # Ensure these names match keys in `extracted_features` dictionary
    model_input_feature_names = [
        # From EDA1 Hybrid Model Inputs, ensure MalwareFeatureExtractor provides these
        'duration', 'beaconing_score', 'error_persistence_rate', # error_persistence_rate from AFE
        'malicious_tool_freq', # From AFE
        'hex_code_anomaly_score', # From AFE, renamed from hex_anomaly
        'log_decoupling_index', # From AFE (placeholder)
        'residual_correlation_score', # From AFE (placeholder)
        'non_repeating_chain_length', # From AFE
        'pair_entropy', # From AFE
        'predictability_score', # From AFE
        'cross_variable_influence', # From AFE
        # Additional useful session aggregates
        'session_event_count', 'session_unique_event_ids_count',
        'prop_error', 'prop_warning', # From AFE
        'session_total_malicious_keywords', 'session_avg_malicious_keywords_per_event',
        'session_total_hex_codes', 'session_avg_hex_codes_per_event'
        # Add other features as they are solidified
    ]

    # Select only existing features and reorder, then add essential ID and label cols
    final_feature_columns = ['session_id', 'session_start_time', 'session_end_time']
    for col_name in model_input_feature_names:
        if col_name in session_features_df.columns:
            final_feature_columns.append(col_name)
        else:
            logger.warning(f"Feature '{col_name}' not found in generated session features. Will be missing from output.")
            # Optionally, add it with default value: session_features_df[col_name] = 0.0

    final_feature_columns.append('session_label') # Add label at the end

    # Ensure all columns in final_feature_columns exist in session_features_df before reindexing
    existing_final_cols = [col for col in final_feature_columns if col in session_features_df.columns]
    session_features_df = session_features_df[existing_final_cols]


    logger.info(f"Successfully generated session-level features. Final shape: {session_features_df.shape}")
    return session_features_df

if __name__ == "__main__":
    # This block allows running this script to generate features.
    # It expects the output from the data_processing stage.

    # Path to the combined, cleaned, event-level log data with session_id and initial lexical features
    # This file is the output of your data_processing scripts (log_parser.py, log_parser_malware.py)
    # after they are combined.
    input_processed_event_logs_path = os.path.join(PROJECT_ROOT_BF, 'data', 'processed', 'all_cleaned_event_logs.csv') # Adjust name as needed
    output_session_features_path = os.path.join(PROJECT_ROOT_BF, 'data', 'processed', 'session_features_for_modeling.csv')

    if not os.path.exists(input_processed_event_logs_path):
        logger.error(f"Input file not found: {input_processed_event_logs_path}")
        logger.error("Please run data_processing scripts first to generate the cleaned event-level data with session IDs.")
    else:
        logger.info(f"Loading cleaned event logs from: {input_processed_event_logs_path}")
        try:
            # Specify dtypes for problematic columns if known, or use low_memory=False
            cleaned_event_df = pd.read_csv(
                input_processed_event_logs_path,
                parse_dates=['timestamp'],
                low_memory=False
            )

            # --- Crucial: Pre-calculate top_pairs for predictability score ---
            # This should be done on a representative (normal or mixed) dataset and saved/loaded.
            # For this example, we'll derive it from the input data, but ideally, it's from a stable training corpus.
            logger.info("Calculating top event pairs for predictability score (ideally pre-calculate from training set)...")
            temp_event_sequences = cleaned_event_df.groupby('session_id')['event_id'].apply(list)
            all_pairs = []
            for seq in temp_event_sequences:
                if len(seq) >= 2:
                    all_pairs.extend(list(zip(seq[:-1], seq[1:])))

            top_200_pairs_set = set()
            if all_pairs:
                pair_counts = Counter(all_pairs)
                top_200_pairs_set = set(p[0] for p in pair_counts.most_common(200))
                logger.info(f"Derived {len(top_200_pairs_set)} top pairs for predictability calculation.")
            else:
                logger.warning("No pairs found to derive top_pairs for predictability.")
            # --- End top_pairs calculation ---


            session_features_final_df = generate_session_level_features(
                cleaned_event_df,
                top_pairs_for_predictability=top_200_pairs_set
                # hex_whitelist_set can be passed if different from default
            )

            if not session_features_final_df.empty:
                os.makedirs(os.path.dirname(output_session_features_path), exist_ok=True)
                session_features_final_df.to_csv(output_session_features_path, index=False)
                logger.info(f"Session features for modeling saved to: {output_session_features_path}")

                print("\n--- Sample of Final Session Features ---")
                display(session_features_final_df.head())
                print("\n--- Final Session Features Info ---")
                session_features_final_df.info()
            else:
                logger.warning("No session features were generated by the pipeline.")

        except Exception as e:
            logger.error(f"An error occurred during the feature building pipeline: {e}", exc_info=True)