# Multi-stage build for EdgeMind Platform
# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install core requirements (skip problematic ones for now)
RUN pip install --upgrade pip && \
    pip install \
    together>=0.2.11 \
    streamlit>=1.28.0 \
    pandas>=2.1.0 \
    plotly>=5.17.0 \
    requests>=2.31.0 \
    python-dotenv>=1.0.0 \
    beautifulsoup4>=4.12.0 \
    lxml>=4.9.0 \
    aiohttp>=3.9.0 \
    validators>=0.22.0 \
    loguru>=0.7.0 \
    psutil>=5.9.0 \
    numpy>=1.24.0 \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.5.0 \
    httpx>=0.25.0 \
    duckduckgo-search>=3.9.0 \
    rich>=13.6.0 \
    tqdm>=4.66.0

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 edgemind && \
    mkdir -p /app /app/models /app/data && \
    chown -R edgemind:edgemind /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=edgemind:edgemind . .

# Switch to non-root user
USER edgemind

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - Streamlit
CMD ["streamlit", "run", "web/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]