from os import getenv

import uvicorn

from api.api import app


if __name__ == "__main__":
	uvicorn.run(
		app,
		host=getenv("APP_HOST", "127.0.0.1"),
		port=int(getenv("APP_PORT", "8000")),
	)
