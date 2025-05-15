# Use Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Ensure the model directory is also copied

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "fastapiapp:app", "--host", "0.0.0.0", "--port", "8000"]
