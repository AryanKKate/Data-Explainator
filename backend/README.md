# Data Explainator

AI-powered data analysis and ML pipeline orchestration system that automatically analyzes datasets and builds predictive models using multi-agent LLM coordination.

## Features

- **Automated Data Analysis**: Comprehensive profiling and quality checks
- **Intelligent Intent Detection**: Automatically identifies ML task type from natural language
- **Smart Feature Engineering**: Semantic-aware feature extraction and preprocessing
- **Multi-Model Training**: Trains and compares multiple ML models
- **Production-Ready**: Full error handling, logging, and validation

## Architecture

```
Data Input
    ↓
Intent Detection (LLM) → Schema Analysis
    ↓
Data Cleaning & Validation
    ↓
Semantic Analysis → Feature Engineering
    ↓
Model Selection (LLM)
    ↓
Multi-Model Training
    ↓
Results & Validation
```

## Quick Start

### Prerequisites
- Python 3.10+
- Groq API key for LLM
- pip

### Installation

1. **Clone repository**
```bash
cd backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### Running

**Development Server (FastAPI)**
```bash
python -m uvicorn api_server:app --reload
```

**Command Line (Main Script)**
```bash
python main.py
```

**Run Tests**
```bash
pytest tests/ -v --cov=
```

## Configuration

### Environment Variables (.env)

```
# LLM Configuration
GROQ_API_KEY=your_api_key_here
MODEL_NAME=mixtral-8x7b-32768

# Behavior
LLM_TEMPERATURE=0.2
DEBUG=False
LOG_LEVEL=INFO

# Constraints
MAX_FILE_SIZE_MB=100
TEST_MODE=False
```

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| GROQ_API_KEY | - | Required Groq API key |
| MODEL_NAME | mixtral-8x7b-32768 | LLM model to use |
| LLM_TEMPERATURE | 0.2 | LLM temperature (0-1) |
| MAX_FILE_SIZE_MB | 100 | Max file size limit |
| LOG_LEVEL | INFO | Logging level |
| DEBUG | False | Enable debug mode |

## API Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy", "timestamp": "..."}
```

### Upload & Analyze
```
POST /upload-analyze
Content-Type: multipart/form-data

Form Data:
  - file: <binary file>
  - query: "Predict housing prices..."

Response: {
  "session_id": "uuid",
  "status": "processing",
  "profile": { shape, columns, dtypes, missing, ... },
  "timestamp": "..."
}
```

### Full Analysis
```
POST /analyze
Content-Type: application/json

Body: {
  "user_query": "Predict housing prices using price as target"
}

Response: {
  "session_id": "uuid",
  "status": "completed",
  "task_type": "regression",
  "target_column": "price",
  "timestamp": "..."
}
```

## Project Structure

```
backend/
├── agents/                          # ML agents
│   ├── intent_agent.py             # Task detection
│   ├── planning_agent.py           # Strategy planning
│   ├── semantic_agent.py           # Semantic analysis
│   ├── schema_agent.py             # Schema detection
│   ├── feature_agent.py            # Feature engineering
│   ├── model_selection.py          # Model selection
│   ├── training_agent.py           # Model training
│   └── validation_agent.py         # Validation
├── services/                        # Data services
│   ├── load_data.py                # File loading
│   ├── clean_data.py               # Data cleaning
│   └── profile_data.py             # Data profiling
├── tools/                          # Utilities
│   └── preprocessing_tools.py      # Preprocessing
├── utils/                          # Utilities
│   └── llm.py                      # LLM configuration
├── graphs/                         # LangGraph workflows
│   └── analyst_graph.py            # Main orchestration
├── tests/                          # Unit tests
│   ├── test_load_data.py
│   └── test_clean_data.py
├── config.py                       # Configuration
├── exceptions.py                   # Custom exceptions
├── logger_config.py                # Logging setup
├── requirements.txt                # Dependencies
├── .env.example                    # Environment template
├── api_server.py                   # FastAPI server
└── main.py                         # CLI entry point
```

## Error Handling

The application uses custom exceptions for specific error scenarios:

```python
from exceptions import (
    FileOperationError,
    DataValidationError,
    LLMError,
    ModelTrainingError,
)

try:
    df = DataLoader.load("file.csv")
except FileOperationError as e:
    logger.error(f"File operation failed: {e}")
```

## Logging

Logs are written to console and file (if configured):

```
2024-01-15 10:30:45 - myapp.services - INFO - Data loaded successfully: (1000, 25)
2024-01-15 10:30:46 - myapp.agents - DEBUG - Task detected: regression (confidence: 0.95)
```

## Testing

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_load_data.py -v
```

### Run with coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

### Available Test Files
- `tests/test_load_data.py` - File loading tests
- `tests/test_clean_data.py` - Data cleaning tests

## Performance Considerations

- **Large Datasets (>100K rows)**: Uses LightGBM for faster training
- **Small Datasets (<1K rows)**: Uses simpler models like Random Forest
- **Missing Values**: Multiple filling strategies (median, mode, forward fill)
- **Outliers**: IQR and Z-score detection with capping

## Known Limitations

1. **LLM Latency**: Intent detection uses LLM (0.5-2 seconds)
2. **File Size**: Default limit 100MB (configurable)
3. **Memory**: Large datasets loaded entirely into memory
4. **Categorical**: Supports up to 50 unique categories per column

## Security

- Input validation on all API endpoints
- File size limits to prevent DoS
- API key validation at startup
- CORS enabled for frontend integration
- No credentials logged

## Contributing

1. Follow PEP 8 style guide
2. Add tests for new features
3. Update requirements.txt with version pins
4. Document functions with docstrings
5. Use type hints throughout

## Troubleshooting

### LLM Connection Error
```
LLMError: Failed to initialize LLM: Connection timeout
```
**Solution**: Check GROQ_API_KEY is valid and network is connected

### Out of Memory
```
MemoryError: Unable to allocate 2.00 GiB
```
**Solution**: Reduce MAX_FILE_SIZE_MB or use data sampling

### Import Errors
```
ModuleNotFoundError: No module named 'catboost'
```
**Solution**: Run `pip install -r requirements.txt`

## Support

For issues and questions:
1. Check the documentation
2. Review error logs
3. Check existing issues
4. Create detailed issue report

## License

Proprietary - All rights reserved

## Roadmap

- [ ] Distributed training with Dask
- [ ] Real-time model monitoring
- [ ] AutoML optimization
- [ ] Explainability reports (SHAP)
- [ ] Database backend support
- [ ] Kubernetes deployment templates
