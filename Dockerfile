# Use an official lightweight Python runtime
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for some Python packages (like building C extensions)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Hugging Face Spaces expects the app to run on port 7860 by default
EXPOSE 7860

# Command to run the FastAPI server
CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "7860"]
