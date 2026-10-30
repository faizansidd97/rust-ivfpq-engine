FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    python3 \
    python3-numpy \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain 1.78
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.78.0 \
    && chmod -R a+w /usr/local/rustup /usr/local/cargo

# Create unprivileged agent user and configure isolated permissions
RUN useradd -m -u 1000 -s /bin/bash agent && \
    mkdir -p /workspace /logs/verifier /root/tests /tests && \
    chmod 700 /logs/verifier /root/tests && \
    chown -R agent:agent /workspace

WORKDIR /workspace
COPY --chown=agent:agent workspace/ /workspace/

# Pre-build dependencies under agent user so Cargo registry is cached offline
USER agent
RUN cargo build --release
USER root
