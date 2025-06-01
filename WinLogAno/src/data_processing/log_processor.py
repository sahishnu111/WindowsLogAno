import os
import pandas as pd
import re
from datetime import timedelta


class LogProcessor:
    MALICIOUS_KEYWORDS = ['powershell', 'rundll32', 'wmi', 'remote', 'invoke',
                          'encodedcommand', 'iex', 'scriptblock', 'bypass',
                          'executionpolicy', 'hidden', 'obfuscated']
    HEX_PATTERN = r'0x[0-9a-fA-F]{4,}'
    BEACON_THRESHOLD = timedelta(minutes=5)

    # Standard columns required for processing
    REQUIRED_COLUMNS = ['timecreated', 'id', 'leveldisplayname',
                        'providername', 'message', 'logname']

    def __init__(self, dataset_type='normal'):
        self.dataset_type = dataset_type
        self.keywords = self.MALICIOUS_KEYWORDS + (
            ['malware', 'ransomware', 'exploit'] if dataset_type == 'malware' else []
        )

    def load_and_clean(self, file_path):
        """Unified loading and cleaning for both normal and malware datasets"""
        # Validate file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Log file {file_path} does not exist")

        # Dataset-specific loading
        if self.dataset_type == 'malware':
            df = self._load_and_clean_malware(file_path)
        else:
            df = self._load_and_clean_normal(file_path)

        # Common cleaning steps
        return self._apply_common_cleaning(df)

    def _load_and_clean_normal(self, file_path):
        """Specialized loading for normal datasets with validation"""
        print(f"Loading normal dataset: {file_path}")
        df = pd.read_csv(file_path)

        # Apply your original cleaning steps
        # Convert critical columns with error handling
        df['TimeCreated'] = pd.to_datetime(df['TimeCreated'], errors='coerce', utc=True)
        df['Id'] = pd.to_numeric(df['Id'], errors='coerce')

        # Drop rows with invalid critical fields
        df = df.dropna(subset=['TimeCreated', 'Id'])

        # Convert ID to integer after cleaning
        df['Id'] = df['Id'].astype(int)

        # Standardize column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        # Clean string columns
        string_cols = ['leveldisplayname', 'providername', 'message', 'logname']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().fillna('N/A')

        return df

    def _load_and_clean_malware(self, file_path):
        """Specialized loading for malware datasets"""
        print(f"Loading malware dataset: {file_path}")
        df = pd.read_csv(file_path)

        # Column mapping for malware datasets
        column_map = {
            'SystemTime': 'TimeCreated',
            'EventID': 'Id',
            'Level': 'LevelDisplayName',
            'ProviderName': 'ProviderName',
            'Description': 'Message',
            'Channel': 'LogName'
        }

        # Validate required columns
        missing_cols = [col for col in column_map.keys() if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Malware dataset missing columns: {missing_cols}")

        # Apply transformations
        df = df[list(column_map.keys())].copy()
        df.rename(columns=column_map, inplace=True)

        # Standardize column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        # Map numeric levels to names
        level_name_map = {
            0: 'Information', 1: 'Critical', 2: 'Error',
            3: 'Warning', 4: 'Information', 5: 'Verbose'
        }
        df['leveldisplayname'] = pd.to_numeric(
            df['leveldisplayname'], errors='coerce'
        ).map(level_name_map).fillna('Unknown')

        return df

    def _apply_common_cleaning(self, df):
        """Common cleaning operations for all datasets"""
        # Validate required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")

        # Ensure timecreated is proper datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timecreated']):
            df['timecreated'] = pd.to_datetime(df['timecreated'], errors='coerce')

        # Final sorting and reset index
        df = df.sort_values('timecreated').reset_index(drop=True)

        # Dataset-specific enhancements
        if self.dataset_type == 'malware':
            df = self._handle_obfuscated_commands(df)
            df = self._filter_hex_anomalies(df)

        return df

    def extract_features(self, df):
        """Extract comprehensive features for malware detection"""
        df = self._add_session_features(df)
        df = self._add_lexical_features(df)
        df = self._add_temporal_features(df)

        session_meta = self._extract_session_metadata(df)
        sequences = self._generate_event_sequences(df)

        return {
            'event_features': df,
            'session_metadata': session_meta,
            'sequences': sequences
        }

    def _add_session_features(self, df):
        """Add session identifiers and temporal features"""
        time_diff = df['timecreated'].diff()
        df['session_id'] = (time_diff > self.BEACON_THRESHOLD).cumsum()
        df['beacon_interval'] = time_diff.apply(lambda x: x.total_seconds())
        return df

    def _add_lexical_features(self, df):
        """Add lexical features using vectorized operations"""
        msg_lower = df['message'].str.lower()

        # Keyword features
        for keyword in self.keywords:
            df[f'kw_{keyword}'] = msg_lower.str.count(keyword)

        df['malicious_kw_count'] = df[[f'kw_{k}' for k in self.keywords]].sum(axis=1)

        # Hex pattern detection
        df['hex_codes'] = df['message'].str.findall(self.HEX_PATTERN).apply(len)

        # Command obfuscation indicators
        df['high_entropy'] = df['message'].apply(self._calculate_entropy)
        return df

    def _add_temporal_features(self, df):
        """Compute temporal patterns and anomalies"""
        # Session-based features
        session_stats = df.groupby('session_id')['beacon_interval'].agg(['mean', 'std'])
        df = df.merge(session_stats, on='session_id', suffixes=('', '_session'))

        # Error persistence tracking
        df['error_persistence'] = df.groupby('eventid')['timecreated'].diff().dt.total_seconds()
        return df

    def _extract_session_metadata(self, df):
        """Compute session-level metadata"""
        return df.groupby('session_id').agg(
            duration=('timecreated', lambda x: (x.max() - x.min()).total_seconds()),
            start_time=('timecreated', 'min'),
            event_count=('id', 'count'),
            beacon_mean=('beacon_interval', 'mean'),
            beacon_std=('beacon_interval', 'std'),
            malicious_kw_total=('malicious_kw_count', 'sum'),
            hex_codes_total=('hex_codes', 'sum')
        ).reset_index()

    def _generate_event_sequences(self, df):
        """Generate event sequences for sequence analysis"""
        return df.groupby('session_id')['eventid'].apply(list)

    def _handle_obfuscated_commands(self, df):
        """Deobfuscate common malware techniques"""
        # Handle base64 encoded commands
        base64_pattern = r"([A-Za-z0-9+/]{4,}={0,2})"
        df['message'] = df['message'].apply(
            lambda x: re.sub(base64_pattern, self._decode_base64, x)
        )

        # Handle charcode obfuscation
        charcode_pattern = r"charcodes?:?\s*(\d+(?:\s*,\s*\d+)+)"
        df['message'] = df['message'].apply(
            lambda x: re.sub(charcode_pattern, self._decode_charcodes, x)
        )
        return df

    def _filter_hex_anomalies(self, df):
        """Flag suspicious hex patterns"""
        hex_density = df['hex_codes'] / df['message'].str.len()
        df['hex_anomaly'] = (hex_density > 0.1).astype(int)
        return df

    def _calculate_entropy(self, text):
        """Calculate Shannon entropy for obfuscation detection"""
        from collections import Counter
        import math
        counter = Counter(text)
        text_len = len(text)
        return -sum(
            (count / text_len) * math.log2(count / text_len)
            for count in counter.values()
        )

    @staticmethod
    def _decode_base64(match):
        """Helper to decode base64 matches"""
        import base64
        try:
            return base64.b64decode(match.group(1)).decode('utf-8', errors='ignore')
        except:
            return match.group(0)

    @staticmethod
    def _decode_charcodes(match):
        """Helper to decode charcode sequences"""
        codes = [int(c) for c in match.group(1).split(',')]
        return ''.join(chr(c) for c in codes)