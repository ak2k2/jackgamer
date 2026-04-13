FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        jq \
        ripgrep \
        tree \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy matplotlib pillow

RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 agent \
    && echo "agent ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER agent

CMD ["sleep", "infinity"]
