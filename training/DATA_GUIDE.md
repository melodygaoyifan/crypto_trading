# 📊 HMATS Training v3.1 - 数据获取指南

## 📋 数据需求汇总

| 模型 | 数据类型 | 最少 | 推荐 | 说明 |
|------|----------|------|------|------|
| **DT v3.2** | OHLCV | 5,000 行 | 15,000+ 行 | ~208 天 / ~625 天 |
| **DRL v5.5** | OHLCV | 3,000 行 | 10,000+ 行 | ~125 天 / ~416 天 |
| **Sentiment v2.2** | 文本+标签 | 3,000 条 | 10,000+ 条 | 需要情感标签 |

---

## 🔧 快速开始

### 方式 1: 自动下载 (推荐)

```bash
cd hmats_training

# 下载全部数据 (OHLCV + 情感)
python get_data.py --all

# 只下载 OHLCV (从 Binance)
python get_data.py --ohlcv --source binance --years 2

# 只下载 OHLCV (从 Kraken)
python get_data.py --ohlcv --source kraken

# 验证数据
python get_data.py --validate
```

### 方式 2: 生成示例数据 (测试用)

```bash
# 生成 20,000 行 OHLCV + 10,000 条情感数据
python get_data.py --generate-sample

# 输出:
# training_data/raw/BTC_60m.parquet
# training_data/raw/ETH_60m.parquet
# training_data/raw/SOL_60m.parquet
# training_data/sentiment/sample_sentiment.parquet
```

---

## 📥 OHLCV 数据获取

### 方式 A: Binance Data Vision (推荐 - 最全)

**官方网站**: https://data.binance.vision/

手动下载步骤:
1. 访问 https://data.binance.vision/
2. 选择 `Spot` → `Monthly` → `Klines`
3. 选择交易对: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`
4. 选择时间间隔: `1h`
5. 下载需要的月份 (建议 2 年)

```bash
# 下载后解压并转换
python get_data.py --import-csv BTCUSDT-1h-2024-01.csv --asset BTC
```

### 方式 B: Kraken API (免费)

```bash
python get_data.py --ohlcv --source kraken
```

注意: Kraken API 每次只返回 720 条，脚本会自动分批获取。

### 方式 C: CryptoDataDownload (简单)

**官方网站**: https://www.cryptodatadownload.com/data/

1. 选择交易所: Binance / Kraken
2. 选择交易对: BTC-USD, ETH-USD, SOL-USD
3. 选择时间框架: 1 hour
4. 下载 CSV

```bash
# 导入下载的 CSV
python get_data.py --import-csv Binance_BTCUSDT_1h.csv --asset BTC
python get_data.py --import-csv Binance_ETHUSDT_1h.csv --asset ETH
python get_data.py --import-csv Binance_SOLUSDT_1h.csv --asset SOL
```

### 方式 D: Kaggle 数据集

```bash
# 安装 kaggle
pip install kaggle

# 配置 API (需要 Kaggle 账号)
# 1. 登录 kaggle.com
# 2. Account → Create API Token
# 3. 将 kaggle.json 放到 ~/.kaggle/

# 下载数据集
kaggle datasets download -d jorijnsmit/binance-full-history
unzip binance-full-history.zip
```

**推荐数据集**:
- `jorijnsmit/binance-full-history` - Binance 全历史
- `jessevent/all-crypto-currencies` - 多币种历史数据
- `sudalairajkumar/cryptocurrency-historical-prices` - 加密货币历史价格

---

## 📰 情感数据获取

### 方式 A: Kaggle 数据集 (推荐)

```bash
# 加密货币新闻情感
kaggle datasets download -d oliviervha/crypto-news-headlines-sentiment
unzip crypto-news-headlines-sentiment.zip -d training_data/sentiment/

# 比特币推文
kaggle datasets download -d kaushiksuresh147/bitcoin-tweets
unzip bitcoin-tweets.zip -d training_data/sentiment/

# 金融新闻情感 (通用)
kaggle datasets download -d ankurzing/sentiment-analysis-for-financial-news
```

**推荐数据集**:
| 数据集 | 内容 | 大小 |
|--------|------|------|
| `oliviervha/crypto-news-headlines-sentiment` | 加密货币新闻标题+情感 | ~10K |
| `kaushiksuresh147/bitcoin-tweets` | BTC 相关推文 | ~100K |
| `zespsolutions/cryptocurrencies-sentiment-analysis` | 多币种情感 | ~50K |
| `ankurzing/sentiment-analysis-for-financial-news` | 金融新闻 | ~5K |

### 方式 B: CryptoCompare News API (免费)

```python
import requests

url = "https://min-api.cryptocompare.com/data/v2/news/"
params = {
    'categories': 'BTC,ETH,SOL',
    'lang': 'EN',
}
response = requests.get(url, params=params)
news = response.json()['Data']

# 注意: 需要手动标注情感
```

### 方式 C: 手动准备

情感数据格式:
```csv
text,label
"Bitcoin surges to new all-time high",2
"Market crashes amid regulatory fears",0
"Trading volume remains steady",1
```

标签:
- `0` = negative (看跌)
- `1` = neutral (中性)
- `2` = positive (看涨)

---

## 📁 目标文件结构

```
training_data/
├── raw/
│   ├── BTC_60m.parquet    # ≥ 5,000 行
│   ├── ETH_60m.parquet    # ≥ 5,000 行
│   └── SOL_60m.parquet    # ≥ 5,000 行
│
├── sentiment/
│   ├── crypto_news.parquet     # 或 .csv
│   └── sentiment_data.parquet  # ≥ 3,000 条
│
└── processed/
    └── (训练时自动生成)
```

### OHLCV 数据格式

```python
import pandas as pd

df = pd.read_parquet('training_data/raw/SOL_60m.parquet')
print(df.columns)
# ['timestamp', 'open', 'high', 'low', 'close', 'volume']

print(df.head())
#              timestamp    open    high     low   close     volume
# 0  2023-01-01 00:00:00  9.5600  9.6100  9.5500  9.5900  1234567.0
# 1  2023-01-01 01:00:00  9.5900  9.6200  9.5800  9.6100  1345678.0
```

### 情感数据格式

```python
df = pd.read_parquet('training_data/sentiment/sentiment_data.parquet')
print(df.columns)
# ['text', 'label', 'timestamp']  # timestamp 可选

print(df.head())
#                                              text  label
# 0  Bitcoin surges to new all-time high             2
# 1  Market faces uncertainty amid regulation        1
# 2  Major exchange hack causes panic selling        0
```

---

## ✅ 验证数据

```bash
python get_data.py --validate
```

输出示例:
```
============================================================
数据验证
============================================================

--- OHLCV 数据 (DT v3.2, DRL v5.5) ---
  BTC: 17,520 条 (730 天)
    DT v3.2:  ✅ (需要 5,000+)
    DRL v5.5: ✅ (需要 3,000+)
  ETH: 17,520 条 (730 天)
    DT v3.2:  ✅ (需要 5,000+)
    DRL v5.5: ✅ (需要 3,000+)
  SOL: 8,760 条 (365 天)
    DT v3.2:  ✅ (需要 5,000+)
    DRL v5.5: ✅ (需要 3,000+)

--- 情感数据 (Sentiment v2.2) ---
  crypto_news.parquet: 10,000 条 ✅ (需要 3,000+)

============================================================
```

---

## 🚀 数据准备完成后

```bash
# 1. 验证数据
python get_data.py --validate

# 2. 修复模块
python fix_modules.py

# 3. 开始训练
python run_training.py --all

# 或分别训练
python train_tqc.py --data training_data/raw/SOL_60m.parquet --presets TQC_1,TQC_2
python drl/train_decision_transformer_v32.py --data training_data/raw/SOL_60m.parquet
python sentiment/train_sentiment_agent_v22.py --data training_data/sentiment/sentiment_data.parquet
```

---

## ⚠️ 常见问题

### Q: 数据不足怎么办?

```bash
# 生成示例数据先测试
python get_data.py --generate-sample

# 然后替换为真实数据
```

### Q: 导入 CSV 格式错误?

确保 CSV 包含这些列 (名称可以不同，会自动映射):
- 时间: `timestamp`, `date`, `Date`, `time`
- 价格: `open`, `high`, `low`, `close` 或 `Open`, `High`, `Low`, `Close`
- 成交量: `volume` 或 `Volume`

### Q: 情感数据没有标签?

可以使用预训练模型自动标注:
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
result = classifier("Bitcoin surges to new high")
# [{'label': 'positive', 'score': 0.95}]
```

### Q: API 被限制?

- Binance Data Vision 无限制 (直接下载文件)
- Kraken API 限制较宽松
- 建议使用 Binance Data Vision 获取大量数据
