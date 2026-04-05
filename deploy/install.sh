#!/bin/bash
# ================================================================================
# HMATS v5.1.0 - Cloud Deployment Install Script
# ================================================================================
# Purpose: Set up HMATS on a fresh cloud server
# 
# Usage:
#   # Full install (systemd service)
#   sudo ./install.sh
#
#   # Docker install only
#   sudo ./install.sh --docker-only
#
# ================================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

HMATS_VERSION="5.1.0"
HMATS_USER="hmats"
HMATS_GROUP="hmats"

INSTALL_DIR="/opt/hmats"
LOG_DIR="/var/log/hmats"
STATE_DIR="/var/lib/hmats"
CONFIG_DIR="/etc/hmats"

PYTHON_VERSION="3.11"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# =============================================================================
# FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

install_dependencies() {
    log_info "Installing system dependencies..."
    
    apt-get update
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        curl \
        git
}

create_user() {
    log_info "Creating HMATS user..."
    
    if id "${HMATS_USER}" &>/dev/null; then
        log_warn "User ${HMATS_USER} already exists"
    else
        useradd -r -s /bin/false -d ${INSTALL_DIR} ${HMATS_USER}
        log_info "Created user: ${HMATS_USER}"
    fi
}

create_directories() {
    log_info "Creating directories..."
    
    mkdir -p ${INSTALL_DIR}
    mkdir -p ${INSTALL_DIR}/models/current
    mkdir -p ${INSTALL_DIR}/configs
    mkdir -p ${LOG_DIR}
    mkdir -p ${LOG_DIR}/events
    mkdir -p ${LOG_DIR}/proof
    mkdir -p ${STATE_DIR}
    mkdir -p ${CONFIG_DIR}
    
    chown -R ${HMATS_USER}:${HMATS_GROUP} ${INSTALL_DIR}
    chown -R ${HMATS_USER}:${HMATS_GROUP} ${LOG_DIR}
    chown -R ${HMATS_USER}:${HMATS_GROUP} ${STATE_DIR}
    chown -R root:${HMATS_GROUP} ${CONFIG_DIR}
    chmod 750 ${CONFIG_DIR}
    
    log_info "Directories created"
}

copy_application() {
    log_info "Copying application files..."
    
    # Get script directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    SOURCE_DIR="${SCRIPT_DIR}/.."
    
    # Copy Python files
    cp -r ${SOURCE_DIR}/*.py ${INSTALL_DIR}/ 2>/dev/null || true
    cp -r ${SOURCE_DIR}/core ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/agents ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/analytics ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/configs ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/defense ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/drl ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/engine ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/exchange ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/execution ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/infra ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/liquidity ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/market ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/market_analysis ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/orchestration ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/overrides ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/risk ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/signals ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/integration ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/legacy ${INSTALL_DIR}/
    cp -r ${SOURCE_DIR}/data_mgmt ${INSTALL_DIR}/
    
    # Copy model configs (not weights)
    mkdir -p ${INSTALL_DIR}/models/decision_transformer
    mkdir -p ${INSTALL_DIR}/models/regime_classifier
    mkdir -p ${INSTALL_DIR}/models/llm_cache
    cp ${SOURCE_DIR}/models/decision_transformer/config.json ${INSTALL_DIR}/models/decision_transformer/ 2>/dev/null || true
    cp ${SOURCE_DIR}/models/regime_classifier/gmm_config.json ${INSTALL_DIR}/models/regime_classifier/ 2>/dev/null || true
    
    chown -R ${HMATS_USER}:${HMATS_GROUP} ${INSTALL_DIR}
    
    log_info "Application files copied"
}

setup_python_venv() {
    log_info "Setting up Python virtual environment..."
    
    python${PYTHON_VERSION} -m venv ${INSTALL_DIR}/venv
    
    ${INSTALL_DIR}/venv/bin/pip install --upgrade pip
    ${INSTALL_DIR}/venv/bin/pip install -r ${INSTALL_DIR}/requirements-runtime.txt
    
    chown -R ${HMATS_USER}:${HMATS_GROUP} ${INSTALL_DIR}/venv
    
    log_info "Python environment ready"
}

install_systemd() {
    log_info "Installing systemd service..."
    
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    
    cp ${SCRIPT_DIR}/systemd/hmats.service /etc/systemd/system/
    
    # Create env file if not exists
    if [ ! -f ${CONFIG_DIR}/hmats.env ]; then
        cp ${SCRIPT_DIR}/systemd/hmats.env.template ${CONFIG_DIR}/hmats.env
        chmod 600 ${CONFIG_DIR}/hmats.env
        chown root:${HMATS_GROUP} ${CONFIG_DIR}/hmats.env
        log_warn "Created ${CONFIG_DIR}/hmats.env - EDIT THIS FILE WITH YOUR API KEYS!"
    fi
    
    systemctl daemon-reload
    systemctl enable hmats
    
    log_info "Systemd service installed"
}

verify_installation() {
    log_info "Verifying installation..."
    
    # Check directories
    for dir in ${INSTALL_DIR} ${LOG_DIR} ${STATE_DIR}; do
        if [ ! -d "$dir" ]; then
            log_error "Directory missing: $dir"
            return 1
        fi
    done
    
    # Check main.py
    if [ ! -f "${INSTALL_DIR}/main.py" ]; then
        log_error "main.py not found"
        return 1
    fi
    
    # Check venv
    if [ ! -f "${INSTALL_DIR}/venv/bin/python" ]; then
        log_error "Python venv not found"
        return 1
    fi
    
    # Test import
    sudo -u ${HMATS_USER} ${INSTALL_DIR}/venv/bin/python -c "from core.cloud_config import get_cloud_config; print('OK')" || {
        log_error "Python import test failed"
        return 1
    }
    
    log_info "Installation verified successfully"
    return 0
}

print_summary() {
    echo ""
    echo "========================================"
    echo "HMATS v${HMATS_VERSION} Installation Complete"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Edit API credentials:"
    echo "   sudo nano ${CONFIG_DIR}/hmats.env"
    echo ""
    echo "2. Sync your trained models:"
    echo "   ./scripts/sync_model_to_cloud.sh"
    echo ""
    echo "3. Start in paper mode:"
    echo "   sudo systemctl start hmats"
    echo ""
    echo "4. View logs:"
    echo "   sudo journalctl -u hmats -f"
    echo ""
    echo "5. Verify operation:"
    echo "   sudo -u hmats ${INSTALL_DIR}/venv/bin/python ${INSTALL_DIR}/main.py --mode verify"
    echo ""
    echo "========================================"
}

# =============================================================================
# MAIN
# =============================================================================

# Parse arguments
DOCKER_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker-only)
            DOCKER_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--docker-only]"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run installation
check_root

echo "========================================"
echo "HMATS v${HMATS_VERSION} Installation"
echo "========================================"

if $DOCKER_ONLY; then
    log_info "Docker-only installation"
    log_info "Build with: docker build -t hmats:${HMATS_VERSION} ."
    log_info "Run with:   docker-compose up -d"
    exit 0
fi

install_dependencies
create_user
create_directories
copy_application
setup_python_venv
install_systemd
verify_installation

print_summary
