# HMATS Hetzner Cloud 部署指南

## 目录

1. [注册与创建服务器](#1-注册与创建服务器)
2. [SSH 配置与连接](#2-ssh-配置与连接)
3. [服务器初始化](#3-服务器初始化)
4. [部署 HMATS](#4-部署-hmats)
5. [模型文件传输](#5-模型文件传输)
6. [启动与验证](#6-启动与验证)
7. [Systemd 守护进程](#7-systemd-守护进程)
8. [监控与告警](#8-监控与告警)
9. [日常维护](#9-日常维护)
10. [费用与优化](#10-费用与优化)

---

## 1. 注册与创建服务器

### 1.1 注册 Hetzner

1. 访问 https://www.hetzner.com/cloud/
2. 点击 "Sign Up"，用邮箱注册
3. 需要信用卡或 PayPal 验证（不会立即扣费）
4. 注册后进入 Cloud Console: https://console.hetzner.cloud/

### 1.2 创建项目

1. Cloud Console → "New Project" → 命名 `hmats`
2. 进入项目

### 1.3 创建 SSH Key（本地操作）

在你的 Windows 终端（PowerShell）中：

```powershell
# 生成 SSH 密钥对（如果还没有的话）
ssh-keygen -t ed25519 -C "melodygao160@gmail.com" -f $env:USERPROFILE\.ssh\hetzner_hmats

# 查看公钥（下一步要粘贴到 Hetzner）
cat $env:USERPROFILE\.ssh\hetzner_hmats.pub
```

### 1.4 在 Hetzner 添加 SSH Key

1. Cloud Console → 左侧 "Security" → "SSH Keys" → "Add SSH Key"
2. 粘贴上一步的公钥内容
3. 命名为 `hmats-deploy`

### 1.5 创建服务器

1. Cloud Console → "Add Server"
2. 配置如下：

| 选项 | 推荐值 | 说明 |
|------|--------|------|
| **Location** | `Nuremberg (eu-central)` | 德国，离 Kraken 最近 |
| **Image** | `Ubuntu 24.04` | LTS，稳定 |
| **Type** | `CPX22` | 够跑 runtime + dashboard |
| **SSH Key** | 选择 `hmats-deploy` | 刚添加的密钥 |
| **Networking** | 勾选 IPv4 | 需要公网 IP |
| **Name** | `hmats-prod` | 服务器名 |

3. 点击 "Create & Buy Now"
4. 记录分配的 **IP 地址**（例如 `65.21.xxx.xxx`）

> **费用**: CPX21 = €7.49/月（按小时计费 €0.0112/h），可随时删除停止计费

---

## 2. SSH 配置与连接

### 2.1 配置 SSH 快捷方式

编辑 `~/.ssh/config`（Windows: `C:\Users\melod\.ssh\config`）：

```
Host hmats
    HostName 65.21.xxx.xxx    # 替换为你的服务器 IP
    User root
    IdentityFile ~/.ssh/hetzner_hmats
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### 2.2 连接测试

```powershell
ssh hmats
```

首次连接会提示确认 fingerprint，输入 `yes`。

---

## 3. 服务器初始化

以下操作在服务器上执行（SSH 连接后）：

### 3.1 系统更新 + 安全加固

```bash
# 更新系统
apt update && apt upgrade -y

# 创建非 root 用户
adduser hmats --disabled-password --gecos ""
usermod -aG sudo hmats

# 允许 hmats 用户 SSH
mkdir -p /home/hmats/.ssh
cp ~/.ssh/authorized_keys /home/hmats/.ssh/
chown -R hmats:hmats /home/hmats/.ssh
chmod 700 /home/hmats/.ssh
chmod 600 /home/hmats/.ssh/authorized_keys

# 防火墙（只开 SSH + Streamlit dashboard）
ufw allow 22/tcp
ufw allow 8501/tcp    # Streamlit dashboard（可选）
ufw --force enable

# 安装 fail2ban（防暴力破解）
apt install -y fail2ban
systemctl enable fail2ban
```

### 3.2 安装 Docker

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh

# 让 hmats 用户使用 docker
usermod -aG docker hmats

# 验证
docker --version
```

### 3.3 安装 Python（如果不用 Docker 部署）

```bash
# 安装 Python 3.12 + 依赖
apt install -y python3.12 python3.12-venv python3-pip git

# 创建符号链接
update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
```

### 3.4 创建目录结构

```bash
# 切换到 hmats 用户
su - hmats

# 创建目录
mkdir -p ~/hmats/{models,logs,data,backups}
```

---

## 4. 部署 HMATS

有两种方式：**Docker（推荐）** 或 **直接部署**。

### 方式 A: Docker 部署（推荐）

```bash
# 以 hmats 用户操作
su - hmats

# 克隆代码
git clone https://github.com/melodygaoyifan/crypto_trading.git ~/hmats/app
cd ~/hmats/app

# 创建 .env 文件
cp .env.example .env
nano .env    # 填入真实 API keys

# 确保 .env 权限安全
chmod 600 .env

# 更新 Dockerfile 版本号（可选）
# Dockerfile 目前引用的是 v5.1.0，可以直接用，不影响运行

# 构建镜像
docker build -t hmats:6.8.0 .

# 验证模式测试
docker run --rm \
  --env-file .env \
  -v ~/hmats/models:/opt/hmats/models:ro \
  -v ~/hmats/logs:/var/log/hmats \
  -v ~/hmats/data:/var/lib/hmats \
  hmats:6.8.0 --mode verify

# Paper trading
docker run -d --name hmats-paper \
  --restart unless-stopped \
  --env-file .env \
  -v ~/hmats/models:/opt/hmats/models:ro \
  -v ~/hmats/logs:/var/log/hmats \
  -v ~/hmats/data:/var/lib/hmats \
  hmats:6.8.0 --mode paper

# 查看日志
docker logs -f hmats-paper
```

### 方式 B: 直接部署（更灵活，推荐调试期使用）

```bash
su - hmats

# 克隆代码
git clone https://github.com/melodygaoyifan/crypto_trading.git ~/hmats/app
cd ~/hmats/app

# 创建虚拟环境
python -m venv ~/hmats/venv
source ~/hmats/venv/bin/activate

# 安装依赖
pip install -r requirements-runtime.txt

# 创建 .env
cp .env.example .env
nano .env    # 填入真实 API keys
chmod 600 .env

# 验证
python main.py --mode verify

# Paper trading（前台运行，看输出）
python main.py --mode paper
```

---

## 5. 模型文件传输

模型文件没有上传到 GitHub（太大），需要从本地传输。

### 从本地 Windows 传输到服务器

在本地 PowerShell 中执行：

```powershell
# 传输 GMM 模型（必须）
scp -r C:\Users\melod\Downloads\hmats\models\regime_classifier hmats:~/hmats/models/

# 传输 DRL 模型（如果训练完成）
scp -r C:\Users\melod\Downloads\hmats\models\retrained hmats:~/hmats/models/

# 传输 Decision Transformer 模型
scp -r C:\Users\melod\Downloads\hmats\models\decision_transformer hmats:~/hmats/models/

# 传输 DRL 训练数据（如果需要在服务器训练）
# scp -r C:\Users\melod\Downloads\hmats\data\drl_training hmats:~/hmats/data/
```

### 验证模型文件

```bash
# 在服务器上
ssh hmats
ls -la ~/hmats/models/regime_classifier/BTC/
# 应该看到: gmm_model.pkl, scaler.pkl, gmm_config.json
```

---

## 6. 启动与验证

### 6.1 Verify 模式快速检查

```bash
cd ~/hmats/app
source ~/hmats/venv/bin/activate   # 直接部署时
python main.py --mode verify
```

应看到：
- 数据获取成功（Kraken ticker/OHLCV）
- GMM regime 预测正常
- Proof log 输出
- 无 CRITICAL 错误

### 6.2 Paper Trading 启动

```bash
# 用 tmux 保持后台运行
apt install -y tmux    # 如果没装

tmux new -s hmats
cd ~/hmats/app
source ~/hmats/venv/bin/activate
python main.py --mode paper

# Ctrl+B, D 断开 tmux（进程继续运行）
# tmux attach -t hmats 重新连接
```

### 6.3 Dashboard（可选）

```bash
tmux new -s dashboard
cd ~/hmats/app
source ~/hmats/venv/bin/activate
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
# Ctrl+B, D 断开
```

访问 `http://你的IP:8501` 查看 dashboard。

---

## 7. Systemd 守护进程

用 systemd 管理 HMATS，实现开机自启 + 崩溃自动重启。

### 7.1 创建 service 文件

```bash
# 以 root 操作
sudo nano /etc/systemd/system/hmats.service
```

写入：

```ini
[Unit]
Description=HMATS Trading System
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=hmats
Group=hmats
WorkingDirectory=/home/hmats/hmats/app
Environment=PATH=/home/hmats/hmats/venv/bin:/usr/bin
EnvironmentFile=/home/hmats/hmats/app/.env
ExecStart=/home/hmats/hmats/venv/bin/python main.py --mode paper
Restart=on-failure
RestartSec=30
StartLimitBurst=5
StartLimitIntervalSec=300

# 日志
StandardOutput=append:/home/hmats/hmats/logs/hmats_stdout.log
StandardError=append:/home/hmats/hmats/logs/hmats_stderr.log

# 安全加固
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/home/hmats/hmats/logs /home/hmats/hmats/data /home/hmats/hmats/app/data

[Install]
WantedBy=multi-user.target
```

### 7.2 启用服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable hmats
sudo systemctl start hmats

# 查看状态
sudo systemctl status hmats

# 查看日志
journalctl -u hmats -f

# 重启
sudo systemctl restart hmats

# 停止
sudo systemctl stop hmats
```

---

## 8. 监控与告警

### 8.1 简易健康检查脚本

```bash
nano /home/hmats/hmats/health_check.sh
```

```bash
#!/bin/bash
# HMATS 健康检查

LOG_DIR="/home/hmats/hmats/logs"
ALERT_FILE="/tmp/hmats_alert_sent"

# 检查进程是否运行
if ! systemctl is-active --quiet hmats; then
    echo "$(date): HMATS is DOWN!" >> $LOG_DIR/health.log
    # 如果配置了 Telegram，发送告警
    if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
        curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}" \
            -d "text=⚠️ HMATS is DOWN on $(hostname)! $(date)" > /dev/null
    fi
    exit 1
fi

# 检查日志是否有 CRITICAL
RECENT_CRITICAL=$(tail -100 $LOG_DIR/hmats_stderr.log 2>/dev/null | grep -c "CRITICAL")
if [ "$RECENT_CRITICAL" -gt 0 ]; then
    echo "$(date): $RECENT_CRITICAL CRITICAL errors in recent logs" >> $LOG_DIR/health.log
fi

# 检查磁盘空间
DISK_PCT=$(df / | tail -1 | awk '{print $5}' | tr -d '%')
if [ "$DISK_PCT" -gt 85 ]; then
    echo "$(date): Disk usage at ${DISK_PCT}%" >> $LOG_DIR/health.log
fi

echo "$(date): OK" >> $LOG_DIR/health.log
```

```bash
chmod +x /home/hmats/hmats/health_check.sh

# 每 5 分钟检查一次
crontab -e -u hmats
# 添加:
*/5 * * * * /home/hmats/hmats/health_check.sh
```

### 8.2 日志轮转

```bash
sudo nano /etc/logrotate.d/hmats
```

```
/home/hmats/hmats/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

### 8.3 Telegram 告警（推荐）

1. Telegram 上找 `@BotFather`，创建 bot，获取 token
2. 发一条消息给你的 bot
3. 访问 `https://api.telegram.org/bot<TOKEN>/getUpdates` 获取 `chat_id`
4. 在 `.env` 中填入 `TELEGRAM_BOT_TOKEN` 和 `TELEGRAM_CHAT_ID`

---

## 9. 日常维护

### 9.1 更新代码

```bash
ssh hmats
cd ~/hmats/app

# 拉取最新代码
git pull origin main

# 重启服务
sudo systemctl restart hmats

# 确认正常
sudo systemctl status hmats
journalctl -u hmats --since "1 min ago"
```

### 9.2 更新模型

从本地传输新模型后：

```bash
# 在服务器上
sudo systemctl stop hmats

# 备份旧模型
cp -r ~/hmats/models ~/hmats/backups/models_$(date +%Y%m%d)

# 从本地 scp 新模型（在本地执行）
# scp -r models/retrained hmats:~/hmats/models/

# 重启
sudo systemctl start hmats
```

### 9.3 备份

```bash
# 每日备份 cron（在服务器上）
crontab -e -u hmats
# 添加:
0 6 * * * tar czf ~/hmats/backups/data_$(date +\%Y\%m\%d).tar.gz ~/hmats/app/data/ ~/hmats/data/
# 保留 7 天
0 7 * * * find ~/hmats/backups/ -name "data_*.tar.gz" -mtime +7 -delete
```

### 9.4 从本地下载日志/数据（在本地 PowerShell 执行）

```powershell
# 下载最近的日志
scp -r hmats:~/hmats/logs/ C:\Users\melod\Downloads\hmats_cloud_logs\

# 下载 shadow ledger
scp -r hmats:~/hmats/app/data/shadow_ledger/ C:\Users\melod\Downloads\hmats_shadow\
```

---

## 10. 费用与优化

### 10.1 费用明细

| 项目 | 月费 | 说明 |
|------|------|------|
| CPX22 服务器 | 按实际定价 | Nuremberg 数据中心 |
| IPv4 地址 | 含在内 | 已包含 |
| 流量 | 免费 | 20TB/月 出站免费 |
| 快照备份 | ~€0.50 | 可选，按使用量 |
| **总计** | **~€8/月 (~$9)** | |

### 10.2 如果需要更多资源

- **内存不够**: 升级到 CPX31 (4 vCPU, 8GB) — €13.49/月
- **需要 GPU 训练**: 创建临时 GPU 实例，训练完删除。或继续用本地 RTX 5090（推荐）
- **多区域冗余**: 在 Helsinki 开一台备用机（生产阶段再考虑）

### 10.3 省钱技巧

- 按小时计费，测试完不用时可以直接删除服务器
- Snapshot 比持续运行空服务器便宜
- 不需要 dashboard 时关闭 Streamlit 节省内存

---

## 快速参考

```bash
# SSH 连接
ssh hmats

# 查看 HMATS 状态
sudo systemctl status hmats

# 查看实时日志
journalctl -u hmats -f

# 重启
sudo systemctl restart hmats

# 停止
sudo systemctl stop hmats

# 磁盘使用
df -h

# 内存使用
free -h

# 进程
ps aux | grep main.py
```

---

## 从 Paper → Live 切换

当 paper trading 验证通过后：

1. 编辑 systemd service：`ExecStart=... --mode live --confirm-live`
2. 确保 `.env` 中 `HMATS_RUNTIME_MODE=PROD`
3. 确认 Kraken API key 有交易权限
4. `sudo systemctl restart hmats`

> ⚠️ 建议先用 `live_phase1.json`（半仓位，2x 杠杆）运行 7 天，再切 `live_phase2.json`（全仓位，3x）。
