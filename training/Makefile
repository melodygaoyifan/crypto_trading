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
	@echo "  make drl                 DRL v7 (~6-12h)"
	@echo "  make drl-fast            DRL v7 快速"
	@echo "  make sentiment           Sentiment v2.2 (~1-3h)"
	@echo "  make sentiment-ensemble  Sentiment 5模型集成"
	@echo ""
	@echo "🔧 工具:"
	@echo "  make check               检查数据"
	@echo "  make collect             收集数据"
	@echo "  make clean               清理"
	@echo ""

# 完整流程
# [P189] --venue/--fee-side threaded through so `make all DRL_VENUE=coinbase`
# means the same thing as `make drl DRL_VENUE=coinbase`. Without them the
# orchestrator's DRL step silently fell back to its own default while the
# `drl` target honoured the variable — two ways to run the same trainer,
# charging different fees, with nothing saying so.
all:
	$(PYTHON) run_training.py --all --data-dir $(DATA_DIR) --output-dir $(OUTPUT_DIR) \
		--venue $(DRL_VENUE) --fee-side $(DRL_FEE_SIDE)

quick:
	$(PYTHON) run_training.py --all --quick --data-dir $(DATA_DIR) --output-dir $(OUTPUT_DIR) \
		--venue $(DRL_VENUE) --fee-side $(DRL_FEE_SIDE)

# GMM
gmm:
	$(PYTHON) run_training.py --gmm --data-dir $(DATA_DIR) --output-dir $(OUTPUT_DIR)

# DRL
dt:
	$(PYTHON) drl/train_decision_transformer_v32.py

dt-max:
	$(PYTHON) drl/train_decision_transformer_v32.py --batch-size 512 --num-workers 16

# [P186 2026-08-05] These two targets invoked drl/train_drl_v55.py, which does
# not exist and is not in git history under that name — `make drl` had been
# failing with "can't open file" for as long as anyone has run it. They also
# passed only SOL_60m.parquet while v5.5 is documented as Cross-Asset
# BTC/ETH/SOL, so the one thing the target could not do even if the script had
# existed was train the asset set its own help text advertises.
#
# The trainer that exists is train_drl_full.py. It takes one --asset per run
# (three separate walk-forward studies, not one multi-asset env), so these
# loop. Flags follow docs/HMATS_TRAINING_GUIDE_V2.md section 5 (the 2026-02
# E2E guide these flags used to cite is archived — its numbers were P164 leak
# artifacts; see docs/archive/HMATS_E2E_TRAINING_GUIDE_2026-02.md banner).
#
# --venue/--fee-side are explicit rather than defaulted: after P179 the
# training env charges real venue fees, and which venue was charged has to be
# visible in the command that produced the model.
DRL_ASSETS ?= BTC ETH SOL
DRL_VENUE ?= kraken
DRL_FEE_SIDE ?= taker
DRL_TIMESTEPS ?= 2500000
DRL_FAST_TIMESTEPS ?= 100000

drl:
	@for a in $(DRL_ASSETS); do \
		echo "=== DRL v7 $$a ($(DRL_TIMESTEPS) steps, $(DRL_VENUE)/$(DRL_FEE_SIDE)) ==="; \
		$(PYTHON) -X utf8 -u train_drl_full.py --asset $$a --folds 3 \
			--timesteps $(DRL_TIMESTEPS) --extractor lstm_film_a \
			--reward-mode classic --no-progress-bar \
			--lr 3e-5 --buffer-size 2000000 --learning-starts 30000 \
			--n-quantiles 32 --dd-threshold 0.10 --dd-penalty-mult 2.0 \
			--reward-clip 50 --regime-weight-min 0.50 --regime-weight-max 2.0 \
			--venue $(DRL_VENUE) --fee-side $(DRL_FEE_SIDE) \
			--output $(OUTPUT_DIR)/drl_v55 --tag makefile_default || exit 1; \
	done

drl-fast:
	@for a in $(DRL_ASSETS); do \
		echo "=== DRL v7 $$a SMOKE ($(DRL_FAST_TIMESTEPS) steps) ==="; \
		$(PYTHON) -X utf8 -u train_drl_full.py --asset $$a --folds 1 \
			--timesteps $(DRL_FAST_TIMESTEPS) --extractor lstm_film_a \
			--reward-mode classic --no-progress-bar \
			--venue $(DRL_VENUE) --fee-side $(DRL_FEE_SIDE) \
			--output $(OUTPUT_DIR)/drl_v55 --tag makefile_fast || exit 1; \
	done

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
	@$(PYTHON) -X utf8 scripts/check_data_freshness.py || true

# [P266] `collect` RETIRED. It invoked the legacy multi-source
# scripts/collect_data.py, which wrote {ASSET}_60m.parquet into the SAME
# directory as the canonical Binance raw parquets — its Kraken source caps
# at ~720 bars, so one run would have truncated SIX YEARS of canonical
# training data to ~a month (the P186 phantom/landmine target class). The
# canonical fetch chain is `make refresh-data` below.
collect:
	@echo "❌ RETIRED (P266): this target overwrote the canonical 6-year raw"
	@echo "   parquets with a ~720-bar Kraken fetch. Use: make refresh-data"
	@exit 1

# [P266] The canonical data-refresh sequence, in dependency order. Safe to
# re-run any time (every step merges or regenerates deterministically).
# Raw OHLCV comes from Binance MONTHLY archives (up to ~31d lag; the current
# month lands ~day 1-2 of the next). The GMM is deliberately NOT refit here:
# {GMM, parquets, checkpoints} move as ONE versioned set (P215/P253b) — refit
# only as a deliberate, recorded decision. fv2 flow features are built inside
# rebuild_pipeline since P266 (the P253c two-step rule is closed).
refresh-data:
	$(PYTHON) -X utf8 fetch_binance_full.py
	$(PYTHON) -X utf8 scripts/fetch_binance_funding.py
	$(PYTHON) -X utf8 scripts/fetch_coinglass_history.py
	$(PYTHON) -X utf8 scripts/rebuild_pipeline.py --smooth 2 --skip-gmm
	$(PYTHON) -X utf8 scripts/generate_split_manifest.py
	$(PYTHON) -X utf8 scripts/refresh_ohlcv_4h.py
	@echo "✅ refresh-data complete — run 'make check' for freshness status"

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ 清理完成"
