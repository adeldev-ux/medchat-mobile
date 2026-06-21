FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE ${PORT:-5000}

CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 app_api_mongo_v2:app