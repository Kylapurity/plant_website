import os
import uvicorn

# Get the port from the environment variable (Render sets this)
port = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    # Run the FastAPI app defined in api.py
    uvicorn.run("api:app", host="0.0.0.0", port=port)