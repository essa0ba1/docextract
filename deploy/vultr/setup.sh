#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu Vultr VPS for docextract (Docker + Compose).
# Run as root: bash deploy/vultr/setup.sh

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo bash deploy/vultr/setup.sh"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  ufw

# Docker official install script
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker

# Docker Compose plugin (included with modern docker-ce)
docker compose version

# Firewall: SSH + HTTP/S only
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo ""
echo "Docker is ready."
echo "Next steps:"
echo "  1. Clone or upload your repo to /opt/docextract"
echo "  2. cd /opt/docextract && docker compose up -d --build"
echo "  3. Optional HTTPS: see deploy/vultr/README.md"
