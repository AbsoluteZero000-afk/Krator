# Multi-stage Dockerfile for Krator Trading System
# Optimized for production with security best practices and minimal image size

# =============================================================================
# Stage 1: Build Dependencies and TA-Lib
# =============================================================================
FROM python:3.11-slim as builder

# Build arguments
ARG TA_LIB_VERSION=0.4.28
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Download and compile TA-Lib from source
WORKDIR /tmp
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Create virtual environment and install Python dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.24.4 && \
    pip install --no-cache-dir TA-Lib && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Stage 2: Production Runtime
# =============================================================================
FROM python:3.11-slim as production

# Build arguments for metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Add metadata labels
LABEL maintainer="Krator Team" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="krator" \
      org.label-schema.description="Real-time self-healing algorithmic trading system" \
      org.label-schema.url="https://github.com/AbsoluteZero000-afk/Krator" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/AbsoluteZero000-afk/Krator" \
      org.label-schema.vendor="Krator" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy TA-Lib libraries from builder stage
COPY --from=builder /usr/local/lib/libta_lib.* /usr/local/lib/
RUN ldconfig

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r krator && \
    useradd -r -g krator -d /app -s /bin/bash -c "Krator user" krator

# Create application directory and set permissions
WORKDIR /app
RUN mkdir -p /app/logs /app/data /app/temp && \
    chown -R krator:krator /app

# Copy application code
COPY --chown=krator:krator . .

# Create .env file from template if it doesn't exist
RUN if [ ! -f .env ]; then cp .env.example .env; fi

# Set up proper file permissions
RUN chmod +x scripts/*.py && \
    find . -name "*.py" -exec chmod +r {} \;

# Health check script
COPY --chown=krator:krator docker/healthcheck.sh /app/healthcheck.sh
RUN chmod +x /app/healthcheck.sh

# Switch to non-root user
USER krator

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOG_LEVEL=INFO \
    ENVIRONMENT=production

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD ./healthcheck.sh

# Default command (can be overridden)
CMD ["python", "scripts/live_trade.py", "--symbols", "AAPL,MSFT,GOOGL,TSLA", "--timeframe", "1Min", "--paper"]

# =============================================================================
# Stage 3: Development Environment (Optional)
# =============================================================================
FROM production as development

# Switch back to root to install dev dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    procps \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    jupyter \
    ipdb

# Switch back to krator user
USER krator

# Override environment for development
ENV ENVIRONMENT=development \
    DEBUG=true \
    LOG_LEVEL=DEBUG

# Development command
CMD ["python", "scripts/quick_check.py"]

# =============================================================================
# Stage 4: Testing Environment (Optional)
# =============================================================================
FROM development as testing

# Testing-specific environment
ENV ENVIRONMENT=testing \
    TEST_MODE=true

# Default test command
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=core", "--cov=data", "--cov=infra"]
