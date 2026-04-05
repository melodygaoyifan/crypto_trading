# ================================================================================
# HMATS Training Pipeline - Makefile
# ================================================================================

PYTHON = python3
DATA_DIR = ./training_data
OUTPUT_DIR = ./models

.PHONY: help all quick gmm dt drl sentiment check clean

help:
	@echo ""
	@echo "================================================================================"
	@echo "                    HMATS Training Pipeline"
	@echo "================================================================================"
	@echo ""
	@echo "📦 完整流程:"
	@echo "  make all                 完整训练 (~12-24h)"
	@echo "  make quick               快速测试 (~1-2h)"
	@echo ""
	@echo "🔄 单独训练:"
	@echo "  make gmm                 6-Regime GMM (~5min)"
	@echo "  make dt                  Decision Transformer v3.2 (~2-4h)"
	@echo "  make dt-max              DT v3.2 最大配置"
	@echo "  make drl                 DRL v5.5 (~6-12h)"
	@echo "  make drl-fast            DRL v5.5 快速"
	@echo "  make sentiment           Sentiment v2.2 (~1-3h)"
	@echo "  make sentiment-ensemble  Sentiment 5模型集成"
	@echo ""
	@echo "🔧 工具:"
	@echo "  make check               检查数据"
	@echo "  make collect             收集数据"
	@echo "  make clean               清理"
	@echo ""

# 完整流程
all:
	$(PYTHON) run_training.py --all --data-dir $(DATA_DIR) --output-dir $(OUTPUT_DIR)

quick:
	$(PYTHON) run_training.py --all --quick --data-dir $(DATA_DIR) --output-dir $(OUTPUT_DIR)

# GMM
gmm:
	$(PYTHON) run_training.py --gmm --data-dir $(DATA_DIR) --output-dir $(OUTPUT_DIR)

# DRL
dt:
	$(PYTHON) drl/train_decision_transformer_v32.py

dt-max:
	$(PYTHON) drl/train_decision_transformer_v32.py --batch-size 512 --num-workers 16

drl:
	$(PYTHON) drl/train_drl_v55.py --data $(DATA_DIR)/raw/SOL_60m.parquet \
		--gmm-features $(DATA_DIR)/processed/gmm_features.parquet \
		--save-dir $(OUTPUT_DIR)/drl_v55 --preset default

drl-fast:
	$(PYTHON) drl/train_drl_v55.py --data $(DATA_DIR)/raw/SOL_60m.parquet \
		--save-dir $(OUTPUT_DIR)/drl_v55 --preset fast

# Sentiment
sentiment:
	$(PYTHON) sentiment/train_sentiment_agent_v22.py

sentiment-large:
	$(PYTHON) sentiment/train_sentiment_agent_v22.py --model large

sentiment-ensemble:
	$(PYTHON) sentiment/train_sentiment_agent_v22.py --ensemble 5

# 工具
check:
	@echo "📊 检查数据..."
	@test -f $(DATA_DIR)/raw/SOL_60m.parquet && echo "✅ SOL" || echo "❌ SOL 缺失"
	@test -f $(DATA_DIR)/raw/BTC_60m.parquet && echo "✅ BTC" || echo "❌ BTC 缺失"
	@test -f $(DATA_DIR)/raw/ETH_60m.parquet && echo "✅ ETH" || echo "❌ ETH 缺失"

collect:
	$(PYTHON) scripts/collect_data.py --source kraken --pairs BTC ETH SOL --output $(DATA_DIR)/raw

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ 清理完成"
