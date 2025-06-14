"""
HTTP Server for Agent API
FastAPI server providing REST endpoints for agent interactions
"""

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import logging
import time
from datetime import datetime

from core.config_loader import ConfigLoader
from core.model_factory import ModelFactory
from core.tool_factory import ToolFactory
from core.agent_service import AgentService


# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for agent query"""
    query: str = Field(..., description="User query to process", min_length=1)
    agent_type: Optional[str] = Field("advanced", description="Type of agent to use")
    thread_id: Optional[str] = Field(None, description="Thread ID for session management")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters for agent")


class QueryResponse(BaseModel):
    """Response model for agent query"""
    answer: str = Field(..., description="Final answer from the agent")
    logs: List[str] = Field(..., description="Execution logs and intermediate steps")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the execution")


class AgentTypeInfo(BaseModel):
    """Information about an agent type"""
    type: str
    name: str
    description: str
    parameters: Dict[str, Any]


class AgentTypesResponse(BaseModel):
    """Response with available agent types"""
    agent_types: Dict[str, str]
    executor_modes: Dict[str, str]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    tools_count: int
    agent_types: List[str]


# Initialize FastAPI app
app = FastAPI(
    title="AI Agent API",
    description="REST API for AI agents with multiple types and execution modes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for service
agent_service: Optional[AgentService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the agent service on startup"""
    global agent_service
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config()
        
        # Create LLM
        llm = ModelFactory.create_llm(config["llm"])
        
        # Create tools
        tool_factory = ToolFactory()
        tools = tool_factory.create_tools(config["tools"], llm)
        
        # Get recursion limit from config
        recursion_limit = config.get("graph", {}).get("recursion_limit", 50)
        
        # Initialize agent service
        agent_service = AgentService(llm, tools, recursion_limit)
        
        logging.info("Agent service initialized successfully")
        logging.info(f"Available tools: {[tool.name for tool in tools]}")
        logging.info(f"Recursion limit: {recursion_limit}")
        
    except Exception as e:
        logging.error(f"Failed to initialize agent service: {str(e)}")
        raise


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not agent_service:
        raise HTTPException(status_code=503, detail="Agent service not initialized")
    
    return HealthResponse(
        status="healthy",
        tools_count=len(agent_service.tools),
        agent_types=list(agent_service.agent_types.keys())
    )


@app.get("/agents", response_model=AgentTypesResponse)
async def get_agent_types():
    """Get available agent types and modes"""
    if not agent_service:
        raise HTTPException(status_code=503, detail="Agent service not initialized")
    
    return AgentTypesResponse(
        agent_types=agent_service.get_available_agent_types(),
        executor_modes=agent_service.get_available_executor_modes()
    )


@app.get("/agents/{agent_type}", response_model=AgentTypeInfo)
async def get_agent_info(agent_type: str):
    """Get information about specific agent type"""
    if not agent_service:
        raise HTTPException(status_code=503, detail="Agent service not initialized")
    
    info = agent_service.get_agent_info(agent_type)
    
    if "error" in info:
        raise HTTPException(status_code=404, detail=info["error"])
    
    return AgentTypeInfo(**info)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a query with the specified agent"""
    if not agent_service:
        raise HTTPException(status_code=503, detail="Agent service not initialized")
    
    try:
        # Process the query
        result = agent_service.process_query(
            query=request.query,
            agent_type=request.agent_type,
            thread_id=request.thread_id,
            **request.parameters
        )
        
        # Add background task for cleanup if needed
        # background_tasks.add_task(cleanup_task, request.thread_id)
        
        return QueryResponse(**result)
        
    except ValueError as e:
        # Handle invalid agent type or parameters
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # Handle unexpected errors
        logging.error(f"Unexpected error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/query/stream")
async def process_query_stream(request: QueryRequest):
    """Process a query with streaming response (future implementation)"""
    # TODO: Implement streaming response for real-time updates
    raise HTTPException(status_code=501, detail="Streaming not yet implemented")


# Optional: Background tasks
def cleanup_task(thread_id: str):
    """Background task for cleanup"""
    logging.info(f"Cleanup task for thread: {thread_id}")
    # Add any cleanup logic here


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logging.error(f"Global exception: {str(exc)}")
    return HTTPException(status_code=500, detail="Internal server error")


def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Run the HTTP server"""
    
    # Setup logging
    log_level = "debug" if debug else "info"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print(f"🚀 Starting AI Agent HTTP Server")
    print(f"📍 Host: {host}")
    print(f"�� Port: {port}")
    print(f"🐛 Debug: {debug}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔄 Alternative docs: http://{host}:{port}/redoc")
    print("=" * 60)
    
    # Run server
    uvicorn.run(
        "http_server:app",
        host=host,
        port=port,
        reload=debug,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agent HTTP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, debug=args.debug) 