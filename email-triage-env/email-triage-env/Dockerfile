FROM python:3.11-slim

LABEL maintainer="hackathon-submission"
LABEL description="Email Triage OpenEnv — Real-world email classification environment"
LABEL tags="openenv,email,triage,rl,agent"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py .
COPY environment.py .
COPY server.py .
COPY inference.py .
COPY openenv.yaml .
COPY data/ ./data/

# Ensure data directory exists
RUN mkdir -p data && touch data/__init__.py

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Environment variables (overridden at runtime)
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV OPENAI_API_KEY=""
ENV ENV_BASE_URL="http://localhost:7860"
ENV PORT=7860

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
