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

# Set up a non-root user (Hugging Face Spaces best practice)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy the rest of your application code and give ownership to the user
COPY --chown=user . $HOME/app
WORKDIR $HOME/app

# Pre-download the FastEmbed models during the Docker Build phase
# This bakes the 1.5GB models into the image so they never download again!
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; TextEmbedding(model_name='BAAI/bge-large-en-v1.5'); SparseTextEmbedding(model_name='prithivida/Splade_PP_en_v1')"

# Hugging Face Spaces expects the app to run on port 7860 by default
EXPOSE 7860

# Command to run the FastAPI server
CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "7860"]
