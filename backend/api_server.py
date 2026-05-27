"""
FastAPI application for Data Explainator.
Provides REST API endpoints for data analysis.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
import logging
import tempfile
import os
from datetime import datetime
import uuid

from config import config
from logger_config import setup_logging, get_logger
from services.load_data import DataLoader
from services.clean_data import DataCleaner
from services.profile_data import DataProfiler
from services.analysis_service import QueryAnalysisService
from graphs.analyst_graph import app as analyst_app
from exceptions import DataExplainterException

# Setup logging
setup_logging(
    level=config.log_level,
    debug=config.debug
)

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Data Explainator API",
    description="AI-powered data analysis and ML pipeline",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AnalysisRequest(BaseModel):
    """Request model for data analysis."""
    
    user_query: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="User's data analysis query"
    )
    
    @validator("user_query")
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class ProfileResponse(BaseModel):
    """Response model for data profile."""
    
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    missing: Dict[str, int]
    memory_usage_mb: float


class AnalysisResponse(BaseModel):
    """Response model for analysis."""
    
    session_id: str
    status: str
    task_type: Optional[str] = None
    target_column: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    timestamp: str


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str
    details: Optional[str] = None
    timestamp: str


@app.get("/", tags=["UI"], include_in_schema=False)
async def ui_home() -> FileResponse:
    """Serve minimal test UI."""
    ui_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    return FileResponse(ui_path)


# API Endpoints

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Dict: Health status
    """
    logger.debug("Health check called")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.post(
    "/upload-analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Upload and analyze data",
)
async def upload_and_analyze(
    file: UploadFile = File(...),
    query: str = Query(..., min_length=5, description="Analysis query"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> Dict[str, Any]:
    """
    Upload a file and start analysis.
    
    Args:
        file: Data file (CSV, Excel, JSON)
        query: Analysis query
        background_tasks: Background task manager
        
    Returns:
        AnalysisResponse: Analysis metadata and initial profile
        
    Raises:
        HTTPException: If validation or processing fails
    """
    session_id = str(uuid.uuid4())
    logger.info(f"Starting analysis session: {session_id}")
    
    try:
        # Validate query
        if not query or len(query.strip()) < 5:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 5 characters"
            )
        
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"{session_id}_{file.filename}")
        
        try:
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
            logger.info(f"File saved: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error saving file: {str(e)}"
            )
        
        # Load and validate data
        try:
            df = DataLoader.load(file_path)
            logger.info(f"Data loaded successfully: {df.shape}")
        except DataExplainterException as e:
            logger.error(f"Data loading failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Data loading failed: {str(e)}"
            )
        
        # Profile data
        try:
            profile = DataProfiler.profile(df)
            logger.info("Data profiling complete")
        except Exception as e:
            logger.error(f"Profiling failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Profiling failed: {str(e)}"
            )
        
        return {
            "session_id": session_id,
            "status": "processing",
            "profile": profile,
            "timestamp": datetime.now().isoformat(),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.post("/analyze-complete", tags=["Analysis"], summary="Upload dataset and run query-driven analysis")
async def analyze_complete(
    file: UploadFile = File(...),
    query: str = Query(..., min_length=5, description="Analysis query"),
) -> Dict[str, Any]:
    """Upload a dataset and run descriptive/diagnostic/predictive/prescriptive analysis based on query."""
    session_id = str(uuid.uuid4())
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"{session_id}_{file.filename}")

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        df = DataLoader.load(file_path)
        cleaned_df = DataCleaner.clean(df)
        analysis = QueryAnalysisService.analyze(cleaned_df, query)

        return {
            "session_id": session_id,
            "status": "completed",
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
        }
    except DataExplainterException as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Unexpected analyze_complete error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                logger.warning("Could not remove temporary file: %s", file_path)


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Analyze data with query",
)
async def analyze_data(request: AnalysisRequest) -> Dict[str, Any]:
    """
    Run full analysis pipeline on uploaded data.
    
    Args:
        request: Analysis request with query
        
    Returns:
        Dict: Analysis results
    """
    session_id = str(uuid.uuid4())
    logger.info(f"Starting full analysis: {session_id}")
    
    try:
        # This would integrate with the analyst_graph
        # For now, return structured response
        return {
            "session_id": session_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/info", tags=["Info"])
async def get_info() -> Dict[str, Any]:
    """
    Get API information.
    
    Returns:
        Dict: API info
    """
    return {
        "name": "Data Explainator",
        "version": "1.0.0",
        "description": "AI-powered data analysis and ML pipeline",
        "endpoints": {
            "upload_analyze": "/upload-analyze",
            "analyze": "/analyze",
            "health": "/health",
        },
    }


@app.exception_handler(DataExplainterException)
async def data_explainter_exception_handler(request, exc: DataExplainterException):
    """Handle custom application exceptions."""
    logger.error(f"Application error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
