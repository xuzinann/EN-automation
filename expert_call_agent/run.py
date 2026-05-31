import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=8888, reload=True)
