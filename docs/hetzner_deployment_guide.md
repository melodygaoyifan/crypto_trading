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

# 防火墙（只开 SSH）
# [FIXED 2026-08-07] 不要开 8501：生产拓扑只暴露 127.0.0.1:8080（hmats-api，
# 见 docker-compose.hetzner.yml），刻意不对外。把 Streamlit 端口开到公网
# 是对活跃交易系统的安全回退。需要看 dashboard 用 SSH 隧道：
#   ssh -L 8501:127.0.0.1:8501 hmats
ufw allow 22/tcp
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

> **[P190 2026-08-06] 生产部署的唯一权威是 `scripts/hetzner_deploy.sh`。**
> 它做的事: `cd /home/hmats/hmats/app` → `git pull origin main` → 检查 `.env`
> → `docker compose -f docker-compose.hetzner.yml build` → 把 models 灌进
> `hmats-models` volume → `docker compose ... up -d`。起来的是两个容器:
> **`hmats-engine`** (由 `Dockerfile.engine` 构建) 和 **`hmats-api`**。
> 所有运维文档里的 `docker exec hmats-engine ...` 指的就是它。
>
> 本节原来的"方式 A"是手工 `docker build .` 单容器 —— 构建的是根目录那份
> v5.1.0 的 `Dockerfile`，容器名 `hmats-paper`，既不是线上跑的镜像也不是
> 线上跑的容器名。"方式 B"(venv) 与第 7 节 systemd 是**非 Docker 的历史
> 路径**，线上没有在用；`deploy/systemd/hmats.service` 至今还写着
> `main.py --mode paper`。保留它们作为参考，但不要照着上生产。

### 方式 A: Docker Compose 部署（生产路径）

```bash
# 以 hmats 用户操作
su - hmats

# 克隆代码（APP_DIR 必须是这个路径，hetzner_deploy.sh 写死了）
git clone https://github.com/melodygaoyifan/crypto_trading.git ~/hmats/app
cd ~/hmats/app

# 创建 .env 文件
cp env/.env.template .env
nano .env    # 填入真实 API keys
chmod 600 .env

# 构建 + 启动（engine + api）
docker compose -f docker-compose.hetzner.yml build
docker compose -f docker-compose.hetzner.yml up -d

# 查看状态与日志
docker compose -f docker-compose.hetzner.yml ps
docker logs -f hmats-engine
```

之后的每次部署都从本地跑一条命令即可，不要在服务器上手工 build：

```bash
# 本地
bash scripts/hetzner_deploy.sh hmats
```

注意 volume：engine 的状态与日志在 `/opt/hmats/data` 和 `/opt/hmats/logs`
（`docker-compose.hetzner.yml` 里挂的是 named volume `hmats-data` /
`hmats-logs`），**不是** `/var/log/hmats` 和 `/var/lib/hmats` —— 那是根目录
那份 v5.1.0 `Dockerfile` 的布局。挂错位置的话容器照常运行，数据却写在镜像
层里，容器重建即丢失。

### 方式 B: 直接部署（历史路径，非生产）

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

> **⚠️ [P191] 只 scp 到 `~/hmats/models/` 不会让引擎看到新模型。** 引擎从
> **命名卷 `hmats-models`**（挂载在容器内 `/opt/hmats/models`，只读）读模型，
> 不是从宿主机目录。scp 之后必须执行 §9.2 的卷同步步骤
> （`docker run --rm -v hmats-models:/models ...`，`hetzner_deploy.sh` Step 4
> 会自动做）。跳过这一步正是 2026-04-22 `models_ready=0` / DRL 卡 SHADOW
> 事故的原因。

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

## 7. Systemd 守护进程（历史路径，线上未使用）

> **[P190] 线上不是 systemd。** 引擎是 `docker-compose.hetzner.yml` 里的
> `hmats-engine`，`restart: unless-stopped` 已经提供开机自启 + 崩溃重启。
> 本节对应的是方式 B 的 venv 安装；仓库里的 `deploy/systemd/hmats.service`
> 至今仍是 `main.py --mode paper`。**事故当中不要用
> `sudo systemctl stop hmats` 停引擎 —— 它什么都不会停。** 对应命令见下表。
>
> | 本节 (systemd) | 线上等价命令 |
> |---|---|
> | `systemctl status hmats` | `cd /home/hmats/hmats/app && docker compose -f docker-compose.hetzner.yml ps` |
> | `systemctl start hmats` | `docker compose -f docker-compose.hetzner.yml up -d` |
> | `systemctl stop hmats` | `docker compose -f docker-compose.hetzner.yml stop hmats-engine` |
> | `systemctl restart hmats` | `docker compose -f docker-compose.hetzner.yml restart hmats-engine` |
> | `journalctl -u hmats -f` | `docker logs -f hmats-engine` |

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
# [P191] 原文用的是 systemd 判断，线上是 docker。对一个不存在的 unit，
# systemd 永远返回 inactive —— 于是健康检查会在引擎完全正常时每分钟报一次 DOWN。
if ! docker inspect -f '{{.State.Running}}' hmats-engine 2>/dev/null | grep -q true; then
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
# [FIXED 2026-08-07] 原来 tail 的 $LOG_DIR/hmats_stderr.log 是 systemd/venv
# 时代的路径，Docker 部署下永远不存在 → grep 永远数出 0（一个不会失败的检查）。
# Docker 下引擎日志在 json-file driver + hmats-logs 卷里，用 docker logs 读：
RECENT_CRITICAL=$(docker logs hmats-engine --since 10m 2>&1 | grep -c "CRITICAL")
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

> **[P191]** 本节原文全部走 systemd。线上是 docker compose，下面已改成
> 实际生效的命令。`scripts/hetzner_deploy.sh` 把这一整套自动化了 ——
> 平时直接在本地跑 `bash scripts/hetzner_deploy.sh hmats` 即可。

```bash
ssh hmats
cd /home/hmats/hmats/app

# 拉取最新代码
git pull origin main

# 重建镜像并重启（代码变了必须 build，不能只 restart）
docker compose -f docker-compose.hetzner.yml build hmats-engine
docker compose -f docker-compose.hetzner.yml up -d

# 确认正常
docker compose -f docker-compose.hetzner.yml ps
docker logs --since 1m hmats-engine
```

### 9.2 更新模型

从本地传输新模型后：

> **[P191]** 引擎从**命名卷** `hmats-models`（只读挂载到
> `/opt/hmats/models`）读模型，不是从 `~/hmats/models` 直接读。只把文件 scp 到
> `~/hmats/models/` 而不同步进卷，引擎看到的还是旧模型 —— 这正是 2026-04-22
> 那次 `models_ready=0`、DRL 卡在 SHADOW 的成因（见 compose 文件里的注释）。

```bash
# 从本地 scp 新模型（在本地执行）
# scp -r models/retrained hmats:~/hmats/models/

# 在服务器上
cd /home/hmats/hmats/app
docker compose -f docker-compose.hetzner.yml stop hmats-engine

# 备份旧模型
cp -r ~/hmats/models ~/hmats/backups/models_$(date +%Y%m%d)

# 同步进命名卷（与 scripts/hetzner_deploy.sh 第 4 步同一条命令）
docker volume create hmats-models 2>/dev/null || true
docker run --rm -v hmats-models:/models -v /home/hmats/hmats/models:/src:ro alpine \
    sh -c "cp -r /src/* /models/ 2>/dev/null || true"

# 重启
docker compose -f docker-compose.hetzner.yml up -d hmats-engine
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

# 查看 HMATS 状态  [P191] 线上是 docker，不是 systemd
cd /home/hmats/hmats/app
docker compose -f docker-compose.hetzner.yml ps

# 查看实时日志
docker logs -f hmats-engine

# 重启
docker compose -f docker-compose.hetzner.yml restart hmats-engine

# 停止
docker compose -f docker-compose.hetzner.yml stop hmats-engine

# 磁盘使用
df -h

# 内存使用
free -h

# 容器资源占用
docker stats --no-stream hmats-engine hmats-api
```

---

## 从 Paper → Live 切换

当 paper trading 验证通过后：

> **[P191]** 模式写在 `docker-compose.hetzner.yml` 的 `command:` 里，不在 systemd
> unit 里。**线上当前已经是 live**：
> `command: ["--mode", "live", "--config", "configs/live_high_risk.json", "--confirm-live"]`。

1. 编辑 `docker-compose.hetzner.yml` 中 `hmats-engine` 的 `command:` 数组
2. 确保 `.env` 中 `HMATS_RUNTIME_MODE=PROD`
3. 确认 Kraken API key 有交易权限
4. `docker compose -f docker-compose.hetzner.yml up -d hmats-engine`（改 `command:`
   后必须 `up -d` 重建容器，`restart` 不会应用新的 command）

> ⚠️ 建议先用 `live_phase1.json`（半仓位，2x 杠杆）运行 7 天，再切 `live_phase2.json`（全仓位，3x）。

---

## 附录: CPX31 快速部署流程 (2026-04 更新)

适用于用户已选 CPX31 (4 vCPU / 8 GB / €14 mo), 直接切 live 模式。

### 1. 本地准备 (laptop)
```bash
# 在 laptop 上打包同步 (exclude 训练产物 + 日志)
rsync -avz --progress \
  --exclude='.git' --exclude='training/' --exclude='logs/' \
  --exclude='data/live_experiences/' --exclude='data/shadow_ledger/' \
  --exclude='__pycache__' --exclude='*.pyc' \
  /c/Users/melod/Downloads/hmats/ \
  hetzner:~/hmats/
```

### 2. Hetzner 初始化 (一次性)
```bash
ssh hetzner
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker
timedatectl set-ntp true && timedatectl status   # 确认 NTP on
```

### 3. 配置 secrets
```bash
scp .env hetzner:~/hmats/.env
ssh hetzner "chmod 600 ~/hmats/.env"
```

### 4. Seed volumes (models + critical state)
```bash
ssh hetzner
cd ~/hmats
mkdir -p seed/models seed/data
# 从 laptop 同步 models + critical state (在 laptop 跑)
# rsync -avz /c/Users/melod/Downloads/hmats/models/ hetzner:~/hmats/seed/models/
# scp data/drl_promotion_state.json data/tranche_state.json hetzner:~/hmats/seed/data/
bash scripts/seed_volumes.sh
```

### 5. Kraken nonce safety: 停掉 laptop 的 live
在切 cloud 之前必须先 **停 laptop 的 main.py live** (否则 nonce 冲突):
```bash
# 在 laptop
python scripts/launch_live.py stop
```

### 6. Cloud verify → live
```bash
ssh hetzner
cd ~/hmats
# 先 verify 模式 (不下单, 只验证模型 + config 加载)
docker compose -f docker-compose.hetzner.yml run --rm hmats-engine --mode verify
# 无异常后, 切 live
docker compose -f docker-compose.hetzner.yml up -d --build
docker compose -f docker-compose.hetzner.yml logs -f hmats-engine
```

### 7. 验证 checklist (前 30 分钟) — [UPDATED 2026-08-07]
- [ ] `[CCXT] Kraken initialized` (API key 正确 — Kraken 仍是数据源)
- [ ] `[COINBASE-SLEEVE]` / `[COINBASE-MANAGE]` 出现（**真正下单的 venue**；P152 后 Kraken 结构性无仓）
- [ ] `[HEALTH_S1..S12]` 全部 PASS/WARN (无 CRITICAL)
- [ ] ~~`[DRL] FORCE_ACTIVE`~~ **不要检查这一条** — P198 后 DRL 已降级 SHADOW，
      live 配置 `drl.force_active=false` 正是让降级在重启后生效的机制；
      看到 `[DRL_FORCE_ACTIVE]` 反而说明配置错了。预期一条
      `[CUTOVER-IRON-LAW-8]` CRITICAL/启动 属正常（observe-only，P202）
- [ ] `[LIVE_DATA]` 每 ~34s 出现 3 个资产
- [ ] `[ALPHA_GATE]` 阈值合理（Coinbase venue-aware fees 下远低于旧的 Kraken 区间）
- [ ] `[VETO_CHAIN]` 主要 gate 显示 PASS
- [ ] SSH tunnel 后 `curl http://127.0.0.1:8080/health` 返回 200

### 8. 回滚
如果云端 30 分钟内出现 CRITICAL 告警, 或持仓发生异常:
```bash
ssh hmats "cd /home/hmats/hmats/app && docker compose -f docker-compose.hetzner.yml down"
```
Docker volumes 保留所有 state, 无数据损失.

> **[STALE 2026-08-07] "回 laptop 恢复 live" 已不是真实选项** — 系统在
> Coinbase perp sleeve 上有真实持仓，停掉引擎后只剩 venue 上的保护性止损
> (P197/P205) 在管仓位。停机超过一个 tick 周期前，先决定是否需要
> `scripts/coinbase_flatten.py`（操作者手动跑）。

### 已知变更 (本次审计)
- 所有 Windows-only 代码 (live_watchdog.py, launch_live.py, launch_paper.py, health_validator.py) 已加 `sys.platform == 'win32'` 守护, Linux 分支使用 `os.kill(pid, 0)` + `pkill -f` + `pgrep -cf`.
- `docker-compose.hetzner.yml` 默认切 live 模式, 配置改指 `live_high_risk.json`, 资源上限升到 3 vCPU / 6 GB.
- 新增 `scripts/seed_volumes.sh` 负责 models + critical state 初始化.
