from functools import wraps
from flask import request, jsonify
import logging
from config import Config

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    """Security middleware for API endpoints."""
    
    @staticmethod
    def validate_content_length(f):
        """Decorator to validate content length."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            content_length = request.content_length
            if content_length and content_length > Config.MAX_CONTENT_LENGTH:
                logger.warning(f"Request too large: {content_length} bytes")
                return jsonify({
                    'error': 'Request too large',
                    'max_size': Config.MAX_CONTENT_LENGTH
                }), 413
            return f(*args, **kwargs)
        return decorated_function

    @staticmethod
    def validate_file_extension(f):
        """Decorator to validate file extensions."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'file' in request.files:
                file = request.files['file']
                extension = file.filename.split('.')[-1].lower()
                if extension not in Config.ALLOWED_EXTENSIONS:
                    logger.warning(f"Invalid file extension: {extension}")
                    return jsonify({
                        'error': 'File type not allowed',
                        'allowed_types': list(Config.ALLOWED_EXTENSIONS)
                    }), 400
            return f(*args, **kwargs)
        return decorated_function

    @staticmethod
    def rate_limit(limit: int = 100, per: int = 60):
        """Decorator to implement rate limiting."""
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[f"{limit} per {per} seconds"]
        )
        
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                return f(*args, **kwargs)
            return limiter.limit(f"{limit} per {per} seconds")(decorated_function)
        return decorator
