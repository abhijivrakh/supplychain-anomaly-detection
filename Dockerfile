# Dockerfile

# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything into container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port for Streamlit
EXPOSE 8501

# Default command (change to realtime_detection.py if needed)
CMD ["streamlit", "run", "streamlit_app.py"]
