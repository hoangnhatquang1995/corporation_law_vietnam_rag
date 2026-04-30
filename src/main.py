from os import getenv
from pathlib import Path

import uvicorn

from api.api import app


DEFAULT_HOST = "0.0.0.0"


if __name__ == "__main__":
	uvicorn.run(
		app,
		host=getenv("APP_HOST", DEFAULT_HOST),
		port=int(getenv("APP_PORT", "8000")),
	)
