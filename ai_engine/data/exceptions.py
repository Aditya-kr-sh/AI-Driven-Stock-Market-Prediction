"""
Custom exceptions for the Data Ingestion and Management module.
Provides clean error boundaries and structured classifications.
"""

class DataEngineError(Exception):
    """Base exception class for all errors originating in the data module."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


class DownloadError(DataEngineError):
    """Raised when data download fails due to connection loss, API issues, or invalid ticker inputs."""
    pass


class ValidationError(DataEngineError):
    """Raised when data validation constraints (e.g. positive prices, chronological indices) fail."""
    pass


class StorageError(DataEngineError):
    """Raised when reading, writing, or loading dataset files and metadata fails."""
    pass
