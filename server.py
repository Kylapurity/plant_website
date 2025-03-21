import uvicorn
import os
from main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's assigned port or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
