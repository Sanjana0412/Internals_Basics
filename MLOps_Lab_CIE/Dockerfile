FROM python:3.12-slim

WORKDIR /app

# Copy ONLY requirements first — Docker caches this layer
# so re-builds skip pip install if requirements haven't changed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# THEN copy the rest of the code
COPY models/ ./models/
COPY src/predict_cli.py ./src/predict_cli.py

ENTRYPOINT ["python", "src/predict_cli.py"]
