# Use Python 3.10 as base image
FROM python:3.10-slim

# 1) Install curl so the healthcheck can run
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt



# Copy the rest of the application
COPY . .

# Create model directory if it doesn't exist
RUN mkdir -p model

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]