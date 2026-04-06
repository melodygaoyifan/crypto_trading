#!/bin/bash
# =============================================================================
# HMATS v6.8.0 — Hetzner Server Bootstrap
# =============================================================================
# Run ONCE on a fresh Hetzner CPX server as root:
#   ssh root@<IP> 'bash -s' < scripts/hetzner_bootstrap.sh
# =============================================================================
set -euo pipefail

echo "=== HMATS Hetzner Bootstrap ==="

# --- System update ---
apt-get update -qq && apt-get upgrade -y -qq

# --- Create hmats user ---
if ! id hmats &>/dev/null; then
    adduser hmats --disabled-password --gecos ""
    echo "hmats ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/hmats
fi

# Copy SSH keys
mkdir -p /home/hmats/.ssh
cp /root/.ssh/authorized_keys /home/hmats/.ssh/
chown -R hmats:hmats /home/hmats/.ssh
chmod 700 /home/hmats/.ssh
chmod 600 /home/hmats/.ssh/authorized_keys

# --- Firewall ---
ufw allow 22/tcp
ufw allow 8080/tcp   # API (internal only — see nginx for external)
ufw allow 8501/tcp   # Streamlit dashboard (optional)
ufw --force enable

# --- fail2ban ---
apt-get install -y -qq fail2ban
systemctl enable fail2ban

# --- Docker ---
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker hmats
fi

# --- Swap (important for CPX21 with 4GB RAM) ---
if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "vm.swappiness=10" >> /etc/sysctl.conf
    sysctl -p
fi

# --- Directories ---
su - hmats -c "mkdir -p ~/hmats/{app,models,backups}"

# --- Log rotation ---
cat > /etc/logrotate.d/hmats << 'LOGROTATE'
/home/hmats/hmats/app/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
LOGROTATE

echo "=== Bootstrap complete ==="
echo ""
echo "Next steps:"
echo "  1. Add GitHub deploy key (see hetzner_deploy.sh)"
echo "  2. Clone repo"
echo "  3. Place .env"
echo "  4. Upload models"
echo "  5. docker compose up"
