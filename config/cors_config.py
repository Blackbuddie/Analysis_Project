from typing import List, Optional
from config import Config

class CORSConfig:
    """Configuration for CORS settings."""
    
    def __init__(self, allowed_origins: Optional[List[str]] = None):
        self.allowed_origins = allowed_origins or Config.ALLOWED_ORIGINS
        self.methods = ["GET", "POST", "OPTIONS"]
        self.allow_headers = ["Content-Type", "Authorization"]
        self.supports_credentials = True
        
    @property
    def resources(self) -> dict:
        """Return CORS resources configuration."""
        return {
            r"/*": {
                "origins": self.allowed_origins,
                "methods": self.methods,
                "allow_headers": self.allow_headers,
                "supports_credentials": self.supports_credentials
            }
        }

# Default configuration for development
ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
