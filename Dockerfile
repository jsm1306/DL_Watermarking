# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy dependency list and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and templates
COPY app.py .
COPY templates/ ./templates

# Copy model files
# COPY models/ ./models

# Expose Flask default port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
