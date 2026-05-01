FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
	&& apt-get install --no-install-recommends -y libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir --index-url https://pypi.org/simple/ -r requirements.txt --default-timeout=100 -v 2>&1 | tail -100 || true

COPY src/ ./src/

EXPOSE 8000

WORKDIR /app/src

CMD ["python", "main.py"]
