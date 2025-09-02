"""
Custom exceptions for the HieQue framework
"""

class HieQueException(Exception):
    """Base exception class for HieQue framework"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self):
        """Convert exception to dictionary format"""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class DocumentProcessingError(HieQueException):
    """Raised when document processing fails"""
    pass


class RetrievalError(HieQueException):
    """Raised when retrieval operations fail"""
    pass


class EmbeddingError(HieQueException):
    """Raised when embedding generation fails"""
    pass


class ClusteringError(HieQueException):
    """Raised when GMM clustering fails"""
    pass


class OpenAIError(HieQueException):
    """Raised when OpenAI API operations fail"""
    pass


class VectorDatabaseError(HieQueException):
    """Raised when vector database operations fail"""
    pass


class ConfigurationError(HieQueException):
    """Raised when configuration is invalid"""
    pass
    
