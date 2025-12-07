# Project Bicameral - Experiment Log

## Hypothesis

**Contrastive State Steering** can force factual recall in RWKV models without prompt injection.

```
Steering = State(truth) - State(lie)
```

By subtracting the "lie state" from the "truth state", we create a directional vector that pushes probability mass FROM the wrong answer TOWARD the correct answer.

## Test Protocol

1. **No JSON sidecar** - Pure state manipulation only
2. **No context injection** - Simple Q/A prompt format
3. **Multiple boost levels** - Test steering strength from 0.5x to 10x
4. **Three factual questions**:
   - Port number (8088 vs 8080)
   - Restart command (SIGUSR1 vs SIGTERM)
   - Magic bytes (0xDEADBEEF vs 0xCAFEBABE)

## Success Criteria

- Baseline (boost=0): Model likely gets answers wrong
- With steering: Model should get answers correct
- If steering improves accuracy, hypothesis is supported

---

# Experiment Results

