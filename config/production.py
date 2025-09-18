# config/production.py
@dataclass
class ProductionConfig:
    # Rate limiting
    max_requests_per_hour: int = 100
    max_file_size_mb: int = 50
    
    # Security
    allowed_file_types: List[str] = ['.csv', '.xlsx']
    sanitize_column_names: bool = True
    
    # Performance
    max_processing_time_seconds: int = 300
    enable_caching: bool = True
    
    # Monitoring  
    log_level: str = "INFO"
    enable_metrics: bool = True