# src/features/advanced_features.py
"""
Advanced feature engineering for malware detection based on EDA insights.

This module implements the critical malware detection features identified through
comprehensive EDA analysis, including temporal beaconing, error persistence,
sequence analysis, and systemic anomaly detection.
"""

import pandas as pd
import numpy as np
from scipy import fft, stats
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, List, Tuple, Optional, Any, Set
import re
from collections import Counter, defaultdict  # Added defaultdict
import logging

logger = logging.getLogger(__name__)

# Default constants, can be overridden in __init__
DEFAULT_MALICIOUS_KEYWORDS = [
    'powershell', 'rundll32', 'wmi', 'remote', 'cmd.exe', 'cmd',
    'regsvr32', 'mshta', 'certutil', 'bitsadmin', 'wscript', 'cscript'
                                                             'schtasks', 'script', 'execution', 'bypass', 'injection',
    'registry',
    'mimikatz', 'cobaltstrike', 'empire', 'meterpreter', 'payload', 'exploit',
    'shellcode', 'obfuscate', 'uacme', 'proxy', 'dll', 'sysmon', 'whoami',
    'cryptographic', 'beacon', 'c2', 'lateral', 'brute'
]
DEFAULT_HEX_PATTERN = re.compile(r'0x[0-9a-fA-F]{4,}')  # Compile for efficiency
DEFAULT_HEX_WHITELIST = {
    '0x00000000', '0x00000001', '0x000003e9', '0x000001f4',  # Common benign codes
    '0xffffffff'
}


class AdvancedFeatureEngineer:
    def __init__(self, top_pairs: Optional[Set[Tuple[Any, Any]]] = None,
                 hex_whitelist: Optional[Set[str]] = None,
                 malicious_keywords: Optional[List[str]] = None):
        self.malicious_keywords = malicious_keywords if malicious_keywords is not None else DEFAULT_MALICIOUS_KEYWORDS
        self.hex_pattern = DEFAULT_HEX_PATTERN  # Use the compiled pattern
        self.hex_whitelist = hex_whitelist if hex_whitelist is not None else DEFAULT_HEX_WHITELIST
        self.top_pairs = top_pairs if top_pairs is not None else set()  # For predictability

        self.keyword_vectorizer = CountVectorizer(
            vocabulary=self.malicious_keywords,
            binary=False,  # Count occurrences, not just presence
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b"  # Default token pattern
        )

        # Statistical thresholds from EDA (can be updated via a config)
        self.thresholds = {
            'duration_critical': 300,
            'beaconing_critical': 15,  # This was an alert trigger, not a feature calc threshold
            'error_persistence_rate_threshold': 0.5,  # For Error->Error transitions > 50%
            'malicious_freq_threshold': 5,  # per 1k events for alerts
            'predictability_score_threshold': 0.5,
            'pair_entropy_threshold': 2.5,
            'chain_length_threshold': 3,
            'log_decoupling_threshold': 0.3
        }
        # Baseline correlations for decoupling analysis
        self.baseline_correlations = {
            ('Security', 'System'): 0.85,  # Example, should be derived from normal data
            ('Security', 'Application'): 0.72,
            ('System', 'Application'): 0.68
        }

    def compute_error_persistence_rate(self, session_logs_df: pd.DataFrame) -> float:
        """Calculate P(Error | Error) transition rate for event levels in a session."""
        if session_logs_df.empty or 'level' not in session_logs_df.columns or len(session_logs_df) < 2:
            return 0.0

        levels = session_logs_df['level'].astype(str).tolist()
        error_to_error_transitions = 0
        total_from_error_transitions = 0

        for i in range(len(levels) - 1):
            if levels[i].lower() == 'error':
                total_from_error_transitions += 1
                if levels[i + 1].lower() == 'error':
                    error_to_error_transitions += 1

        return error_to_error_transitions / total_from_error_transitions if total_from_error_transitions > 0 else 0.0

    def compute_beaconing_score(self, session_timestamps: pd.Series,
                                resample_freq_seconds: int = 60,  # e.g., 60 for 1 minute
                                freq_range_hz: Tuple[float, float] = (0.001, 0.01)) -> float:
        """FFT-based beaconing detection. Timestamps should be pd.Series of datetimes."""
        if not isinstance(session_timestamps, pd.Series) or session_timestamps.empty or len(session_timestamps) < 10:
            return 0.0

        timestamps = pd.to_datetime(session_timestamps, errors='coerce').dropna().sort_values()
        if len(timestamps) < 10:
            return 0.0

        try:
            # Create time series: count events in each interval
            time_series = pd.Series(1, index=timestamps).resample(f'{resample_freq_seconds}S').count()

            if len(time_series) < 10 or time_series.nunique() == 1:  # Need variance for FFT
                return 0.0

            fft_values = time_series.fillna(0).values
            # Normalize for more stable FFT magnitudes if desired, though max power is often enough
            # fft_values = (fft_values - np.mean(fft_values)) / (np.std(fft_values) + 1e-9) 

            n = len(fft_values)
            # Positive frequencies, d is sample spacing in seconds
            freqs_hz = np.fft.rfftfreq(n, d=resample_freq_seconds)[1:]  # Exclude DC
            power_spectrum = np.abs(np.fft.rfft(fft_values)[1:]) ** 2  # Corresponding power

            if len(freqs_hz) == 0: return 0.0

            mask = (freqs_hz >= freq_range_hz[0]) & (freqs_hz <= freq_range_hz[1])
            relevant_power = power_spectrum[mask]

            return float(np.max(relevant_power)) if relevant_power.size > 0 else 0.0

        except Exception as e:
            logger.warning(f"Beaconing score calculation failed for session: {e}")
            return 0.0

    def compute_malicious_tool_frequency(self, session_messages: pd.Series) -> float:
        """Count occurrences of security-critical keywords per 1000 messages in a session."""
        if session_messages.empty: return 0.0

        text_corpus = session_messages.fillna('').astype(str).tolist()
        if not text_corpus: return 0.0

        try:
            keyword_matrix = self.keyword_vectorizer.fit_transform(text_corpus)  # Fit_transform for each call
            total_keyword_occurrences = keyword_matrix.sum()
            num_messages = len(text_corpus)
            return (total_keyword_occurrences / num_messages) * 1000 if num_messages > 0 else 0.0
        except Exception as e:
            logger.warning(f"Malicious tool frequency calculation failed: {e}")
            return 0.0

    def compute_hex_code_anomaly_score(self, session_messages: pd.Series) -> int:
        """Counts number of suspicious (non-whitelisted) hex codes in session messages."""
        if session_messages.empty: return 0

        all_hex_codes_in_session = set()
        for message in session_messages.fillna('').astype(str):
            found = self.hex_pattern.findall(message.lower())  # Use compiled pattern
            all_hex_codes_in_session.update(found)

        suspicious_hex_codes = [h for h in all_hex_codes_in_session if h not in self.hex_whitelist]
        return len(suspicious_hex_codes)  # Returns count of unique suspicious hex codes

    def compute_chain_metrics(self, session_event_ids: pd.Series) -> Tuple[int, float]:
        """Non-Repeating Chain Length and Pair Entropy for a session's event IDs."""
        if session_event_ids.empty or len(session_event_ids) < 2:
            return 0, 0.0

        event_sequence = session_event_ids.tolist()
        max_chain = 0
        if event_sequence:
            current_chain_elements = set()
            current_chain_length = 0
            for event in event_sequence:
                if event not in current_chain_elements:
                    current_chain_elements.add(event)
                    current_chain_length += 1
                else:  # Repetition found
                    max_chain = max(max_chain, current_chain_length)
                    current_chain_elements = {event}  # Start new chain
                    current_chain_length = 1
            max_chain = max(max_chain, current_chain_length)  # Check last chain

        pairs = list(zip(event_sequence[:-1], event_sequence[1:]))
        pair_entropy_val = 0.0
        if pairs:
            pair_counts_obj = Counter(pairs)
            counts = np.array(list(pair_counts_obj.values()))
            pair_entropy_val = stats.entropy(counts, base=2)

        return max_chain, float(pair_entropy_val)

    def compute_predictability_score(self, session_event_ids: pd.Series) -> float:
        """Computes predictability based on coverage by pre-defined top_pairs."""
        if not self.top_pairs or session_event_ids.empty or len(session_event_ids) < 2:
            # If no top_pairs defined or sequence too short, predictability is low (or undefined)
            return 0.0

        event_sequence = session_event_ids.tolist()
        pairs_in_session = list(zip(event_sequence[:-1], event_sequence[1:]))
        if not pairs_in_session:
            return 0.0  # Or 1.0 if no transitions means perfectly predictable in a way

        covered_pairs_count = sum(1 for pair in pairs_in_session if pair in self.top_pairs)
        return float(covered_pairs_count / len(pairs_in_session))

    def compute_log_decoupling_index(self, session_df: pd.DataFrame,
                                     source_col='source', event_id_col='event_id') -> float:
        """Placeholder for Log Decoupling Index. Needs full implementation."""
        logger.warning(
            "compute_log_decoupling_index is a placeholder and needs full implementation using baseline correlations.")
        # Example logic:
        # 1. Identify events from 'Security' and 'System' sources in session_df
        # 2. Calculate some measure of their co-occurrence or correlation in this session.
        #    This could be a simple count of (Security_event, System_event) pairs within small time windows,
        #    or correlation of their counts over sub-windows.
        # 3. Compare to self.baseline_correlations[('Security', 'System')]
        # 4. Calculate % drop. Repeat for other pairs and average or take max drop.
        return 0.0  # Placeholder

    def compute_residual_correlation_score(self, session_df: pd.DataFrame) -> float:
        """Placeholder for Residual Correlation Score. Needs VAR model residuals."""
        logger.warning("compute_residual_correlation_score is a placeholder and needs VAR model residuals.")
        return 0.0  # Placeholder

    def compute_cross_variable_influence(self, session_df: pd.DataFrame) -> float:
        """Computes cross-variable influence using Cramér's V for Level vs Source."""
        if len(session_df) < 5 or 'level' not in session_df.columns or 'source' not in session_df.columns:
            return 0.0
        try:
            contingency = pd.crosstab(session_df['level'], session_df['source'])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2: return 0.0
            chi2, _, _, _ = stats.chi2_contingency(contingency, correction=False)
            n = contingency.sum().sum()
            if n == 0 or min(contingency.shape) <= 1: return 0.0
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            return float(cramers_v)
        except Exception as e:
            logger.warning(f"Cross-variable influence calculation failed: {e}")
            return 0.0


class MalwareFeatureExtractor:
    """High-level extractor using AdvancedFeatureEngineer and providing risk assessment."""

    def __init__(self, top_pairs: Optional[Set[Tuple[Any, Any]]] = None,
                 hex_whitelist: Optional[Set[str]] = None,
                 malicious_keywords: Optional[List[str]] = None):
        self.feature_engineer = AdvancedFeatureEngineer(top_pairs, hex_whitelist, malicious_keywords)
        # TTP mapping from your LLM explainer idea
        self.ttp_mapping = {
            'malicious_tool_freq': ['T1059.001', 'T1059.003'],  # Command and Scripting Interpreter
            'beaconing_score': ['T1071', 'T1573'],  # Application Layer Protocol, Encrypted Channel
            'error_persistence_rate': ['T1499.004'],  # Endpoint Denial of Service: System Exhaustion
            'non_repeating_chain_length': ['T1027'],  # Obfuscated Files or Information
            'hex_code_anomaly_score': ['T1027'],
            'log_decoupling': ['T1562.001']  # Impair Defenses: Disable or Modify Tools
        }

    def _compute_session_duration(self, session_logs_df: pd.DataFrame) -> float:
        """Helper to compute session duration."""
        if session_logs_df.empty or 'timestamp' not in session_logs_df.columns or len(session_logs_df) < 1:
            return 0.0
        timestamps = pd.to_datetime(session_logs_df['timestamp'], errors='coerce').dropna()
        if len(timestamps) < 2: return 0.0
        return (timestamps.max() - timestamps.min()).total_seconds()

    def extract_all_features(self, session_logs_df: pd.DataFrame) -> Dict[str, Any]:
        """Extracts all defined features for a single session DataFrame."""
        if not isinstance(session_logs_df, pd.DataFrame) or session_logs_df.empty:
            logger.warning("extract_all_features received empty or invalid session_logs_df.")
            # Return a dictionary with default zero values for all expected features
            return {
                'duration': 0.0, 'error_persistence_rate': 0.0, 'beaconing_score': 0.0,
                'malicious_tool_freq': 0.0, 'hex_code_anomaly_score': 0,
                'non_repeating_chain_length': 0, 'pair_entropy': 0.0,
                'predictability_score': 0.0, 'log_decoupling_index': 0.0,
                'residual_correlation_score': 0.0, 'cross_variable_influence': 0.0,
                'prop_error': 0.0, 'prop_warning': 0.0,
                # Add event_ids if needed for sequence input to model
                'event_ids_for_sequence': []
            }

        features = {}
        features['duration'] = self._compute_session_duration(session_logs_df)

        # Features from AdvancedFeatureEngineer
        features['error_persistence_rate'] = self.feature_engineer.compute_error_persistence_rate(session_logs_df)
        features['beaconing_score'] = self.feature_engineer.compute_beaconing_score(session_logs_df['timestamp'])
        features['malicious_tool_freq'] = self.feature_engineer.compute_malicious_tool_frequency(
            session_logs_df['message'])
        features['hex_code_anomaly_score'] = self.feature_engineer.compute_hex_code_anomaly_score(
            session_logs_df['message'])

        chain_len, pair_ent = self.feature_engineer.compute_chain_metrics(session_logs_df['event_id'])
        features['non_repeating_chain_length'] = chain_len
        features['pair_entropy'] = pair_ent

        features['predictability_score'] = self.feature_engineer.compute_predictability_score(
            session_logs_df['event_id'])

        features['log_decoupling_index'] = self.feature_engineer.compute_log_decoupling_index(session_logs_df)
        features['residual_correlation_score'] = self.feature_engineer.compute_residual_correlation_score(
            session_logs_df)  # Placeholder
        features['cross_variable_influence'] = self.feature_engineer.compute_cross_variable_influence(session_logs_df)

        # Proportion features (can be calculated here or in AdvancedFeatureEngineer)
        if 'level' in session_logs_df.columns:
            level_counts = session_logs_df['level'].value_counts(normalize=True)
            features['prop_error'] = level_counts.get('Error', 0.0)
            features['prop_warning'] = level_counts.get('Warning', 0.0)
        else:
            features['prop_error'] = 0.0
            features['prop_warning'] = 0.0

        # Store event IDs for sequence input to model (if model needs it directly)
        features['event_ids_for_sequence'] = session_logs_df[
            'event_id'].tolist() if 'event_id' in session_logs_df.columns else []

        return features

    def get_risk_assessment_from_features(self, extracted_features: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level based on pre-extracted features and EDA-validated thresholds."""
        # This method uses the thresholds from AdvancedFeatureEngineer instance
        thresholds = self.feature_engineer.thresholds
        risk_assessment = {'risk_level': 'LOW', 'triggered_rules': [], 'mitre_ttps': []}

        # Apply rules based on "EDA1" Critical Alert Triggers
        # RED: (Beaconing_Score > 15) + (Malicious_Tool_Freq > 10)
        if extracted_features.get('beaconing_score', 0) > thresholds.get('beaconing_critical', 15) and \
                extracted_features.get('malicious_tool_freq', 0) > thresholds.get('malicious_freq',
                                                                                  5):  # EDA1 used 10 for critical alert
            risk_assessment['risk_level'] = 'CRITICAL'
            risk_assessment['triggered_rules'].append('High Beaconing & High Malicious Tools')
            risk_assessment['mitre_ttps'].extend(
                self.ttp_mapping.get('beaconing_score', []) + self.ttp_mapping.get('malicious_tool_freq', []))

        # ORANGE: (Predictability_Score < 0.4) + (Pair_Entropy > 2.5)
        elif extracted_features.get('predictability_score', 1.0) < thresholds.get('predictability', 0.5) and \
                extracted_features.get('pair_entropy', 0) > thresholds.get('pair_entropy', 2.5):
            risk_assessment['risk_level'] = 'HIGH'
            risk_assessment['triggered_rules'].append('Low Predictability & High Entropy')
            risk_assessment['mitre_ttps'].extend(
                self.ttp_mapping.get('non_repeating_chain_length', []))  # Maps to obfuscation

        # YELLOW: Log_Decoupling_Index > 30% + Hex_Code_Anomaly > threshold
        # Assuming hex_code_anomaly_score is a count, threshold could be e.g. > 0
        elif extracted_features.get('log_decoupling_index', 0) > thresholds.get('log_decoupling_threshold',
                                                                                0.3) * 100 and \
                extracted_features.get('hex_code_anomaly_score', 0) > 0:  # If any suspicious hex code found
            risk_assessment['risk_level'] = 'MEDIUM'
            risk_assessment['triggered_rules'].append('Log Decoupling & Hex Anomaly')
            risk_assessment['mitre_ttps'].extend(
                self.ttp_mapping.get('log_decoupling', []) + self.ttp_mapping.get('hex_code_anomaly_score', []))

        # Error Persistence (from EDA1: Error->Error > 50%)
        elif extracted_features.get('error_persistence_rate', 0) > thresholds.get('error_persistence_rate_threshold',
                                                                                  0.5):
            if risk_assessment['risk_level'] not in ['CRITICAL', 'HIGH']: risk_assessment['risk_level'] = 'MEDIUM'
            risk_assessment['triggered_rules'].append('High Error Persistence')
            risk_assessment['mitre_ttps'].extend(self.ttp_mapping.get('error_persistence_rate', []))

        # Remove duplicate TTPs
        risk_assessment['mitre_ttps'] = list(set(risk_assessment['mitre_ttps']))

        return risk_assessment


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    sample_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='min'),
        'level': ['Information'] * 50 + ['Warning'] * 30 + ['Error'] * 20,
        'message': ['Test message'] * 100,
        'event_id': np.random.randint(1000, 1100, size=100)
    })

    feature_extractor = MalwareFeatureExtractor()
    features = feature_extractor.extract_all_features(sample_df)
    risk_assessment = feature_extractor.get_risk_assessment_from_features(features)

    print("Extracted Features:", features)
    print("Risk Assessment:", risk_assessment)
