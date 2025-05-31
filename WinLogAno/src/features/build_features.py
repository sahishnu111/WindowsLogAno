import numpy as np
import pandas as pd
from scipy import fft
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


# Markov Features
def markov_features(log_df):
    """Calculate Markov transition probabilities"""
    transition_counts = defaultdict(lambda: defaultdict(int))

    for _, group in log_df.groupby('session_id'):
        events = group['event_type'].tolist()
        for i in range(len(events) - 1):
            current = events[i]
            next_evt = events[i + 1]
            transition_counts[current][next_evt] += 1

    # Convert to probabilities
    transition_matrix = {}
    for current, next_counts in transition_counts.items():
        total = sum(next_counts.values())
        transition_matrix[current] = {next_evt: count / total
                                      for next_evt, count in next_counts.items()}

    return {
        'error_persistence': transition_matrix['Error']['Error'],
        'warn_to_error': transition_matrix['Warning']['Error'],
        'info_stability': transition_matrix['Information']['Information']
    }


# FFT Features
def fft_features(time_series, sample_rate=1 / 60):
    """Detect periodic patterns using FFT"""
    n = len(time_series)
    yf = fft.fft(time_series)
    power = np.abs(yf[:n // 2]) ** 2
    freqs = sample_rate * np.arange(n // 2) / n
    beacon_score = np.max(power[(freqs > 0.001) & (freqs < 0.01)])
    return beacon_score


# Sequence Features
def sequence_features(log_df):
    """Extract sequence-based features"""
    sequences = log_df.groupby('session_id')['event_id'].apply(list)

    # Pair coverage calculation
    pairs = defaultdict(int)
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pairs[pair] += 1

    total = sum(pairs.values())
    top_200 = sorted(pairs.values(), reverse=True)[:200]
    predictability = sum(top_200) / total

    # Pair entropy
    probs = np.array(list(pairs.values())) / total
    pair_entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # Longest non-repeating chain
    max_chain = max(len(set(seq)) for seq in sequences)

    return predictability, pair_entropy, max_chain


# Keyword Features
def keyword_features(log_texts):
    """Extract security-related keywords"""
    malware_keywords = ['powershell', 'rundll32', 'wmi', 'uacme', 'proxy',
                        'sysmon', 'whoami', 'cryptographic', 'remote']
    counts = [sum(kw in text.lower() for text in log_texts)
              for kw in malware_keywords]
    return sum(counts) / len(log_texts) * 1000  # per 1k logs