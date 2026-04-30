FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
	&& apt-get install --no-install-recommends -y libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY src/ ./src/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

WORKDIR /app/src

CMD ["python", "main.py"]