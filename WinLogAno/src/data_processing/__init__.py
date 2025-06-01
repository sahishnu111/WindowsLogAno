
from .log_processor import LogProcessor
from .log_parser import parse_log as parse_normal_log
from .log_parser_malware import parse_log as parse_malware_log

__all__ = [
    'LogProcessor',
    'parse_normal_log',
    'parse_malware_log'
]