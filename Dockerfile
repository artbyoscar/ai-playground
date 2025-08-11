# Multi-stage build for EdgeMind Platform with High-Performance Kernels
# Stage 1: Kernel Builder
FROM ubuntu:22.04 AS kernel-builder

# Install build tools for kernels
RUN apt-get update && apt-get install -y \
    clang-15 \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy kernel source code
WORKDIR /build
COPY src/kernels/cpu/int4 ./kernels/

# Build high-performance kernels
WORKDIR /build/kernels
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-15 \
    -DCMAKE_CXX_COMPILER=clang++-15 \
    -DINT4_FUSE_BIAS=ON \
    && cmake --build build

# Stage 2: Python Builder
FROM python:3.12-slim AS python-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install core requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    pandas>=2.1.0 \
    streamlit>=1.28.0 \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    psutil>=5.9.0 \
    plotly>=5.17.0 \
    matplotlib>=3.7.0 \
    requests>=2.31.0 \
    python-dotenv>=1.0.0 \
    aiohttp>=3.9.0 \
    loguru>=0.7.0 \
    rich>=13.6.0 \
    tqdm>=4.66.0 \
    && pip cache purge

# Stage 3: Runtime
FROM python:3.12-slim AS runtime

# Install runtime dependencies and security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    curl \
    libgomp1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user and directories
RUN useradd -m -u 1000 edgemind && \
    mkdir -p /app /app/models /app/data /app/kernels /app/tools && \
    chown -R edgemind:edgemind /app

# Copy built kernels from kernel-builder
COPY --from=kernel-builder --chown=edgemind:edgemind /build/kernels/build /tmp/kernels-build
RUN find /tmp/kernels-build -name "*.a" -exec cp {} /app/kernels/ \; 2>/dev/null || true && \
    find /tmp/kernels-build -name "*.so" -exec cp {} /app/kernels/ \; 2>/dev/null || true && \
    rm -rf /tmp/kernels-build

# Copy headers
COPY --from=kernel-builder --chown=edgemind:edgemind /build/kernels /tmp/kernels-src
RUN find /tmp/kernels-src -maxdepth 1 -name "*.h" -exec cp {} /app/kernels/ \; 2>/dev/null || true && \
    find /tmp/kernels-src -maxdepth 1 -name "*.hpp" -exec cp {} /app/kernels/ \; 2>/dev/null || true && \
    rm -rf /tmp/kernels-src

# Copy virtual environment from python-builder
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code (includes edgemind_kernels.py and entrypoint.sh)
COPY --chown=edgemind:edgemind . .

# Ensure entrypoint is executable
RUN chmod +x /app/entrypoint.sh

# Environment variables
ENV PYTHONPATH=/app \
    LD_LIBRARY_PATH=/app/kernels:/app/src/kernels/cpu/int4/build \
    EDGEMIND_KERNELS_PATH=/app/kernels \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    EDGEMIND_PERFORMANCE_MODE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Switch to non-root user
USER edgemind

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${STREAMLIT_SERVER_PORT}/_stcore/health || \
        curl -f http://localhost:8000/health || exit 1

# Labels
LABEL maintainer="EdgeMind Team" \
      version="1.0.0" \
      description="EdgeMind Platform with 125+ GFLOP/s INT8/Q8 Kernels" \
      performance="125+ GFLOP/s on AMD Ryzen 7 8840HS"

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]