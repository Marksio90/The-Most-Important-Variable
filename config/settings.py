"""
Enhanced configuration system with comprehensive settings management.
Simplified structure with better error handling and validation.
"""
import os
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from functools import lru_cache
import json
from datetime import datetime

try:
    from pydantic import BaseModel, Field, validator, ConfigDict
    from pydantic_settings import BaseSettings
    PYDANTIC_V2 = True
except ImportError:
    try:
        # Fallback for older pydantic versions
        from pydantic import BaseModel, Field, validator, BaseSettings
        PYDANTIC_V2 = False
        class ConfigDict:
            pass
    except ImportError:
        # Final fallback - basic dataclass-like behavior
        BaseSettings = object
        BaseModel = object
        Field = lambda default=None, **kwargs: default
        validator = lambda *args, **kwargs: lambda f: f
        ConfigDict = lambda **kwargs: None
        PYDANTIC_V2 = False

# Environment types
class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

# Logging levels
class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# ML Engine options
class MLEngine(str, Enum):
    AUTO = "auto"
    SKLEARN = "sklearn"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"

# Simplified configuration classes - flatter structure
class Settings(BaseSettings):
    """Main application settings with simplified flat structure."""
    
    # Environment and basic app info
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(default=True, description="Debug mode")
    app_name: str = Field(default="TMIV", description="Application name")
    version: str = Field(default="2.0.0", description="Application version")
    
    # Data handling configuration
    data_max_file_size_mb: int = Field(default=500, ge=1, le=2000, description="Maximum file size in MB")
    data_supported_formats: List[str] = Field(
        default=["csv", "xlsx", "json", "parquet"],
        description="Supported file formats"
    )
    data_input_dir: str = Field(default="data", description="Input data directory")
    data_output_dir: str = Field(default="tmiv_out", description="Output directory")
    data_models_dir: str = Field(default="models", description="Saved models directory")
    data_logs_dir: str = Field(default="logs", description="Logs directory")
    data_cache_dir: str = Field(default=".cache", description="Cache directory")
    data_temp_dir: str = Field(default="temp", description="Temporary files directory")
    
    # ML configuration
    ml_default_engine: MLEngine = Field(default=MLEngine.AUTO, description="Default ML engine")
    ml_available_engines: List[str] = Field(
        default=["auto", "sklearn", "lightgbm", "xgboost", "catboost"],
        description="Available ML engines"
    )
    ml_default_test_size: float = Field(default=0.2, ge=0.05, le=0.5, description="Default test split ratio")
    ml_default_cv_folds: int = Field(default=5, ge=2, le=20, description="Default cross-validation folds")
    ml_random_state: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    ml_n_jobs: int = Field(default=-1, description="Number of parallel jobs (-1 for all cores)")
    ml_max_training_time: int = Field(default=1800, ge=60, description="Maximum training time in seconds")
    ml_hyperopt_trials: int = Field(default=100, ge=10, le=1000, description="Hyperparameter optimization trials")
    ml_hyperopt_timeout: int = Field(default=600, ge=60, description="Hyperparameter optimization timeout")
    ml_use_optuna: bool = Field(default=True, description="Use Optuna for hyperparameter optimization")
    ml_max_features: int = Field(default=1000, ge=10, description="Maximum number of features")
    ml_correlation_threshold: float = Field(default=0.95, ge=0.8, le=1.0, description="Feature correlation threshold")
    
    # API configuration
    api_host: str = Field(default="127.0.0.1", description="API host")
    api_port: int = Field(default=8000, ge=1000, le=65535, description="API port")
    api_debug: bool = Field(default=False, description="API debug mode")
    api_secret_key: str = Field(default="dev-secret-key-change-in-production", description="Secret key for sessions")
    api_max_request_size: int = Field(default=100, description="Maximum request size in MB")
    api_rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    api_requests_per_minute: int = Field(default=100, ge=1, description="Requests per minute limit")
    
    # Logging configuration
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_to_file: bool = Field(default=True, description="Enable file logging")
    log_to_console: bool = Field(default=True, description="Enable console logging")
    log_file_max_bytes: int = Field(default=10_000_000, description="Maximum log file size in bytes")
    log_backup_count: int = Field(default=5, description="Number of backup log files")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # Database configuration
    db_sqlite_path: str = Field(default="data/tmiv.db", description="SQLite database path")
    db_pool_size: int = Field(default=5, ge=1, description="Database connection pool size")
    db_pool_timeout: int = Field(default=30, ge=1, description="Connection pool timeout")
    
    # Feature flags - simplified structure
    feature_advanced_preprocessing: bool = Field(default=True, description="Enable advanced preprocessing")
    feature_auto_feature_engineering: bool = Field(default=True, description="Enable auto feature engineering")
    feature_model_interpretability: bool = Field(default=True, description="Enable model interpretability")
    feature_real_time_predictions: bool = Field(default=True, description="Enable real-time predictions")
    feature_batch_processing: bool = Field(default=True, description="Enable batch processing")
    feature_model_monitoring: bool = Field(default=True, description="Enable model monitoring")
    
    # Advanced settings
    enable_async_training: bool = Field(default=True, description="Enable asynchronous model training")
    enable_model_versioning: bool = Field(default=True, description="Enable model versioning")
    enable_experiment_tracking: bool = Field(default=True, description="Enable experiment tracking")
    
    # Configuration for pydantic
    if PYDANTIC_V2:
        model_config = ConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            env_prefix="TMIV_",
            case_sensitive=False,
            validate_assignment=True,
            extra="ignore"  # Ignore extra fields instead of failing
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            env_prefix = "TMIV_"
            case_sensitive = False
            validate_assignment = True
            extra = "ignore"
    
    # Safer validators that won't crash the app
    @validator("version", allow_reuse=True)
    def validate_version_format(cls, v):
        """Validate version format with fallback."""
        try:
            import re
            if not re.match(r'^\d+\.\d+\.\d+', str(v)):
                logging.warning(f"Version {v} doesn't follow semantic versioning, using default")
                return "2.0.0"
            return v
        except Exception:
            return "2.0.0"
    
    @validator("data_supported_formats", allow_reuse=True)
    def validate_supported_formats(cls, v):
        """Ensure we have at least basic formats."""
        if not v:
            return ["csv", "json"]
        valid_formats = ["csv", "json", "xlsx", "parquet", "feather", "pickle"]
        return [fmt for fmt in v if fmt in valid_formats] or ["csv", "json"]
    
    @validator("ml_available_engines", allow_reuse=True) 
    def validate_engines(cls, v):
        """Ensure we have at least basic engines."""
        if not v:
            return ["auto", "sklearn"]
        valid_engines = ["auto", "sklearn", "lightgbm", "xgboost", "catboost"]
        return [eng for eng in v if eng in valid_engines] or ["auto", "sklearn"]
    
    # Helper methods
    def get_feature_flag(self, flag_name: str) -> bool:
        """Get feature flag value safely."""
        attr_name = f"feature_{flag_name}"
        return getattr(self, attr_name, False)
    
    def enable_feature(self, flag_name: str) -> None:
        """Enable a feature flag."""
        attr_name = f"feature_{flag_name}"
        if hasattr(self, attr_name):
            setattr(self, attr_name, True)
    
    def disable_feature(self, flag_name: str) -> None:
        """Disable a feature flag."""
        attr_name = f"feature_{flag_name}"
        if hasattr(self, attr_name):
            setattr(self, attr_name, False)
    
    def get_data_paths(self) -> Dict[str, Path]:
        """Get all data paths as Path objects."""
        return {
            "input": Path(self.data_input_dir),
            "output": Path(self.data_output_dir), 
            "models": Path(self.data_models_dir),
            "logs": Path(self.data_logs_dir),
            "cache": Path(self.data_cache_dir),
            "temp": Path(self.data_temp_dir)
        }
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        try:
            paths = self.get_data_paths()
            for path_name, path in paths.items():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logging.warning(f"Could not create directory {path}: {e}")
        except Exception as e:
            logging.error(f"Error creating directories: {e}")
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        # Simplified - only SQLite for now
        return f"sqlite:///{self.db_sqlite_path}"
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION
    
    def get_log_level(self) -> str:
        """Get logging level as string."""
        return self.log_level.value if hasattr(self.log_level, 'value') else str(self.log_level)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        try:
            if PYDANTIC_V2:
                return self.model_dump()
            else:
                return self.dict()
        except Exception:
            # Fallback for compatibility issues
            result = {}
            for key in dir(self):
                if not key.startswith('_') and not callable(getattr(self, key)):
                    try:
                        value = getattr(self, key)
                        if isinstance(value, (str, int, float, bool, list, dict)):
                            result[key] = value
                        elif hasattr(value, 'value'):  # Enum
                            result[key] = value.value
                        else:
                            result[key] = str(value)
                    except Exception:
                        continue
            return result
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save current configuration to file."""
        try:
            config_dict = self.to_dict()
            
            # Convert Path objects and other complex types to strings
            def convert_complex_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_complex_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_complex_types(item) for item in obj]
                elif isinstance(obj, Path):
                    return str(obj)
                elif hasattr(obj, 'value'):  # Enum
                    return obj.value
                else:
                    return obj
            
            config_dict = convert_complex_types(config_dict)
            
            filepath = Path(filepath)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
                
            logging.info(f"Configuration saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
    
    def apply_environment_overrides(self) -> None:
        """Apply environment-specific overrides safely."""
        try:
            if self.environment == Environment.PRODUCTION:
                # Production overrides
                self.debug = False
                self.api_debug = False
                if self.api_secret_key == "dev-secret-key-change-in-production":
                    logging.warning("Using default secret key in production!")
                
            elif self.environment == Environment.TESTING:
                # Testing overrides
                self.data_output_dir = "test_outputs"
                self.db_sqlite_path = ":memory:"
                self.log_level = LogLevel.WARNING
                
            elif self.environment == Environment.DEVELOPMENT:
                # Development overrides
                self.debug = True
                self.api_debug = True
                
        except Exception as e:
            logging.error(f"Error applying environment overrides: {e}")


# Singleton pattern for settings with caching and error handling
_settings_instance = None
_settings_lock = False

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached application settings with proper error handling.
    
    This function uses LRU cache to ensure settings are loaded only once
    and reused across the application lifecycle.
    """
    global _settings_instance, _settings_lock
    
    if _settings_instance is not None:
        return _settings_instance
    
    if _settings_lock:
        # Prevent infinite recursion during initialization
        return _create_fallback_settings()
    
    _settings_lock = True
    
    try:
        settings = Settings()
        settings.apply_environment_overrides()
        settings.create_directories()
        _settings_instance = settings
        return settings
        
    except Exception as e:
        logging.error(f"Failed to load settings: {e}")
        return _create_fallback_settings()
        
    finally:
        _settings_lock = False

def _create_fallback_settings() -> Settings:
    """Create minimal fallback settings when main initialization fails."""
    try:
        # Create bare minimum settings
        if PYDANTIC_V2 or hasattr(Settings, '__init__'):
            return Settings()
        else:
            # Final fallback for broken pydantic installations
            class FallbackSettings:
                def __init__(self):
                    self.app_name = "TMIV"
                    self.version = "2.0.0"
                    self.environment = Environment.DEVELOPMENT
                    self.debug = True
                    self.data_max_file_size_mb = 200
                    self.data_supported_formats = ["csv", "json"]
                    self.ml_default_engine = MLEngine.AUTO
                    self.log_level = LogLevel.INFO
                
                def get_feature_flag(self, flag_name: str) -> bool:
                    return True
                
                def to_dict(self) -> Dict[str, Any]:
                    return self.__dict__.copy()
                
                def get_data_paths(self) -> Dict[str, Path]:
                    return {
                        "input": Path("data"),
                        "output": Path("tmiv_out"),
                        "models": Path("models"),
                        "logs": Path("logs"),
                        "cache": Path(".cache"),
                        "temp": Path("temp")
                    }
                
                def create_directories(self) -> None:
                    for path in self.get_data_paths().values():
                        try:
                            path.mkdir(parents=True, exist_ok=True)
                        except Exception:
                            pass
            
            return FallbackSettings()
            
    except Exception as e:
        logging.critical(f"Even fallback settings failed: {e}")
        raise RuntimeError("Cannot initialize application settings")

def get_config_summary(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Get a summary of current configuration with error handling."""
    try:
        if settings is None:
            settings = get_settings()
        
        return {
            "environment": getattr(settings, 'environment', 'unknown'),
            "version": getattr(settings, 'version', '2.0.0'),
            "debug": getattr(settings, 'debug', True),
            "app_name": getattr(settings, 'app_name', 'TMIV'),
            "ml_engine": getattr(settings, 'ml_default_engine', 'auto'),
            "api_port": getattr(settings, 'api_port', 8000),
            "log_level": getattr(settings, 'log_level', 'INFO'),
            "config_loaded_at": datetime.now().isoformat(),
            "data_paths": {
                k: str(v) for k, v in settings.get_data_paths().items()
            } if hasattr(settings, 'get_data_paths') else {}
        }
        
    except Exception as e:
        logging.error(f"Failed to generate config summary: {e}")
        return {
            "environment": "unknown",
            "version": "2.0.0",
            "error": str(e),
            "config_loaded_at": datetime.now().isoformat()
        }

def setup_logging(settings: Optional[Settings] = None) -> None:
    """Setup logging based on configuration with error handling."""
    try:
        if settings is None:
            settings = get_settings()
        
        # Create logs directory safely
        try:
            logs_path = Path(getattr(settings, 'data_logs_dir', 'logs'))
            logs_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            logs_path = Path('logs')
            logs_path.mkdir(parents=True, exist_ok=True)
        
        # Basic logging configuration
        log_level = getattr(settings, 'log_level', LogLevel.INFO)
        if hasattr(log_level, 'value'):
            log_level = log_level.value
        
        log_format = getattr(settings, 'log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        handlers = []
        
        # Console handler
        if getattr(settings, 'log_to_console', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(console_handler)
        
        # File handler
        if getattr(settings, 'log_to_file', True):
            try:
                file_handler = logging.handlers.RotatingFileHandler(
                    logs_path / "tmiv.log",
                    maxBytes=getattr(settings, 'log_file_max_bytes', 10_000_000),
                    backupCount=getattr(settings, 'log_backup_count', 5)
                )
                file_handler.setFormatter(logging.Formatter(log_format))
                handlers.append(file_handler)
            except Exception as e:
                logging.warning(f"Could not setup file logging: {e}")
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            handlers=handlers,
            force=True  # Override existing configuration
        )
        
        logging.info("Logging configured successfully")
        
    except Exception as e:
        # Final fallback to basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True
        )
        logging.error(f"Failed to setup advanced logging, using basic configuration: {e}")

def validate_configuration(settings: Optional[Settings] = None) -> Dict[str, List[str]]:
    """Validate configuration and return any issues with error handling."""
    issues = {
        "errors": [],
        "warnings": [],
        "info": []
    }
    
    try:
        if settings is None:
            settings = get_settings()
        
        # Check environment-specific issues
        if getattr(settings, 'environment', None) == Environment.PRODUCTION:
            if getattr(settings, 'debug', True):
                issues["warnings"].append("Debug mode enabled in production")
            
            secret_key = getattr(settings, 'api_secret_key', '')
            if 'dev' in secret_key.lower() or secret_key == 'dev-secret-key-change-in-production':
                issues["errors"].append("Default secret key should not be used in production")
        
        # Check file size limits
        max_size = getattr(settings, 'data_max_file_size_mb', 200)
        if max_size > 1000:
            issues["warnings"].append(f"Very large file size limit: {max_size}MB")
        elif max_size < 10:
            issues["warnings"].append(f"Very small file size limit: {max_size}MB")
        
        # Check supported formats
        formats = getattr(settings, 'data_supported_formats', [])
        if not formats:
            issues["errors"].append("No supported file formats configured")
        elif 'csv' not in formats:
            issues["warnings"].append("CSV format not supported - this may limit functionality")
        
        # Check ML engines
        engines = getattr(settings, 'ml_available_engines', [])
        if not engines:
            issues["errors"].append("No ML engines configured")
        elif 'sklearn' not in engines:
            issues["warnings"].append("sklearn engine not available - this may limit functionality")
        
        issues["info"].append(f"Configuration validation completed for {len(dir(settings))} settings")
        
    except Exception as e:
        issues["errors"].append(f"Configuration validation failed: {e}")
    
    return issues

# Utility function for loading from file with error handling
def load_config_from_file(filepath: Union[str, Path]) -> Settings:
    """Load configuration from JSON file with error handling."""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            logging.warning(f"Config file {filepath} not found, using defaults")
            return get_settings()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Filter out invalid keys to prevent pydantic errors
        if PYDANTIC_V2 or hasattr(Settings, '__fields__'):
            valid_keys = set()
            try:
                # Try to get valid field names
                settings_instance = Settings()
                valid_keys = set(settings_instance.to_dict().keys())
            except Exception:
                # Fallback - allow all keys
                valid_keys = set(config_data.keys())
            
            filtered_data = {k: v for k, v in config_data.items() if k in valid_keys}
        else:
            filtered_data = config_data
        
        return Settings(**filtered_data)
        
    except Exception as e:
        logging.error(f"Failed to load config from {filepath}: {e}")
        return get_settings()

# Export main classes and functions
__all__ = [
    "Settings",
    "Environment", 
    "MLEngine",
    "LogLevel",
    "get_settings",
    "get_config_summary",
    "setup_logging",
    "load_config_from_file",
    "validate_configuration"
]