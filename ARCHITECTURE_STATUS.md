# PSA Architecture Status

## Branch: `experiment/bicameral-state`
## Date: 2025-12-06

---

## 1. Current Stable State (Main Branch)

| Attribute | Value |
|-----------|-------|
| **Architecture** | Hybrid Sidecar |
| **Mechanism** | RWKV State (Vibes) + JSON Context (Facts) |
| **Status** | Functional, O(1) inference |
| **Limitation** | Relies on prompt injection for factual precision |

### How It Works (Main Branch)
```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID SIDECAR                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐      ┌─────────────┐                     │
│   │ Delta State │  +   │ JSON Sidecar│                     │
│   │ (Vibes/Tone)│      │ (Hard Facts)│                     │
│   └──────┬──────┘      └──────┬──────┘                     │
│          │                    │                             │
│          ▼                    ▼                             │
│   ┌─────────────────────────────────────┐                  │
│   │         Prompt Construction          │                  │
│   │  "SYSTEM DATA: Port 8088..."        │                  │
│   │  + User Question                     │                  │
│   └─────────────────────────────────────┘                  │
│                      │                                      │
│                      ▼                                      │
│   ┌─────────────────────────────────────┐                  │
│   │         RWKV Inference               │                  │
│   │   (State = Delta, Context = Facts)   │                  │
│   └─────────────────────────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Test Results (Hybrid Sidecar)
- Port 8088: ✅ Correct at gain=0.0 (via JSON injection)
- Port 8088: ❌ Wrong at gain>0.4 (delta interferes)
- SIGUSR1: ❌ Never recalled correctly
- Magic bytes: ⚠️ Partial at gain=0.8

**Conclusion**: JSON sidecar does the heavy lifting. State deltas add "flavor" but degrade factual accuracy at high gains.

---

## 2. New Experiment Goal (This Branch)

| Attribute | Value |
|-----------|-------|
| **Project Name** | Project Bicameral |
| **Hypothesis** | Contrastive State Steering |
| **Mechanism** | Vector subtraction, not text reading |
| **Goal** | Force factual recall through pure state manipulation |

### The Core Insight

Instead of teaching the model what IS true, we teach it what is TRUE vs what is FALSE, then subtract the lie.

```
Traditional Delta:
    S_learned = process("The port is 8088")
    Delta = S_learned - S_0

    Problem: Model encodes "port", "is", "8088" as general concepts.
             No discriminative signal against "8080".

Contrastive Steering:
    S_truth = process("The port is 8088")
    S_lie   = process("The port is 8080")

    Steering = S_truth - S_lie

    Magic: This vector points FROM "8080" TOWARD "8088".
           It's not about what to remember.
           It's about what direction to push probability.
```

### Mathematical Foundation

```
Let P(token | state) be the probability distribution over next tokens.

When we apply:
    state_new = state_base + α * (S_truth - S_lie)

We are effectively doing:
    P(8088 | state_new) > P(8088 | state_base)
    P(8080 | state_new) < P(8080 | state_base)

The subtraction creates a "gradient" in probability space,
pushing the model away from the lie and toward the truth.
```

### Why This Might Work

1. **RWKV State is Semantic**: The state encodes semantic meaning, not just surface tokens.

2. **Contrastive Learning**: In ML, contrastive losses (SimCLR, CLIP) work by pulling positives together and pushing negatives apart. We're doing the same in state space.

3. **No Prompt Injection**: If this works, we don't need the JSON sidecar. The steering vector does the work at the representational level.

---

## 3. Next Steps (The Plan)

### Step A: Implement Contrastive Delta Computation
- Add `kernel.compute_steering_vector(truth_text, lie_text)`
- Process both texts through model
- Return difference vector: `S_truth - S_lie`

### Step B: Implement Steering Application
- Add `kernel.apply_steering(base_state, steering_delta, multiplier)`
- **Critical**: Do NOT normalize this delta
- Steering vectors are directions, not energy states
- We want to push the model HARD in this direction

### Step C: Add CLI Command
- `psa steer <positive_text> <negative_text> --name <save_name> --boost <multiplier>`
- Example: `psa steer "Port is 8088" "Port is 8080" --name hydra_port --boost 2.0`

### Step D: Run the Definitive Test
1. Create steering vector: "8088" vs "8080"
2. **Disable JSON sidecar** (no text injection)
3. Ask: "What port does Hydra run on?"
4. **Expected**: Model says "8088" purely from state steering

---

## 4. Success Criteria

| Test | Expected Outcome |
|------|------------------|
| Port question (no sidecar) | "8088" |
| Restart question (no sidecar) | "SIGUSR1" or "kill -USR1" |
| Magic bytes (no sidecar) | "0xDE 0xAD 0xBE 0xEF" |

If we achieve >2/3 of these, **contrastive steering is validated**.

---

## 5. Risk Assessment

| Risk | Mitigation |
|------|------------|
| Steering too weak | Increase `multiplier` (boost) |
| Steering too strong | Decrease `multiplier` |
| Model outputs garbage | Apply light normalization as fallback |
| No effect at all | Validate state difference is non-zero |

---

## 6. Experiment Log

| Date | Action | Result |
|------|--------|--------|
| 2025-12-06 | Branch created | - |
| 2025-12-06 | Architecture documented | - |
| | Implementation starts | Pending |

---

*"We do not teach. We steer."*
