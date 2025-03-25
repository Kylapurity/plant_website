"""
Entry point for running the FastAPI application on Render.
"""
import os
import uvicorn
import sys

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Get the port from the environment variable (Render sets this)
port = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    # Run the FastAPI app defined in api.py
    uvicorn.run("src.api:app", host="0.0.0.0", port=port)