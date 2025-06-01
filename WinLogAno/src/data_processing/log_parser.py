from .log_processor import LogProcessor


def parse_log(file_path):
    """
    Parse normal log files with enhanced feature extraction

    :param file_path: Path to normal log file
    :return: Dictionary of extracted features
    """
    processor = LogProcessor(dataset_type='normal')
    df = processor.load_and_clean(file_path)
    return processor.extract_features(df)


