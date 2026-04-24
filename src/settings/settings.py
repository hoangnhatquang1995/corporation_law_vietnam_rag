import os
from dotenv import load_dotenv
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_URL  = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
