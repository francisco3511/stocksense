"""Stocksense package for stock selection"""

try:
    from importlib.metadata import version

    __version__ = version("stocksense")
except ImportError:
    # Package is not installed
    __version__ = "1.0.0"
