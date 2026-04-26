# Smart Beta V1 — A/B Validation Plan

## Phase A: Baseline (current)
- Smart Beta `enabled=false`
- Record: gate_mult, size_mult, scale-in events, PnL, DD, exposure

## Phase B: Observe-Only
- Config: `{"enabled": true, "mode": "observe_only"}`
- Zero decision impact
- Verify: `[SMART_BETA_STATE]` logs every tick, tags match regime events
- Duration: 48-72h minimum

## Phase C: Bounded Influence
- Config: `{"enabled": true, "mode": "bounded_influence"}`
- Only modulates gate/size/confidence/scale-in/short restriction
- Never flips direction, never vetoes

## Metrics
- Total PnL, Max DD, Win Rate, Avg Trade Size
- Alpha gate filter rate (should increase in drift, decrease in trend)
- Scale-in blocked events (should correlate with neutral drift / crowding)
- Short restriction events
- Regime-bucket PnL (should improve in QUIET_ACCUMULATION)
- Rolling beta / downside beta

## Keep/Kill Rules
- If observe_only state quality poor (wrong tags) → fix before Phase C
- If Phase C worsens PnL > 5% or DD > 3% → disable immediately
- Kill: `"enabled": false` in config + restart
