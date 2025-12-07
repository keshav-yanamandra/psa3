#!/usr/bin/env python3
"""Quick test script for PSA v3"""

import sys
sys.path.insert(0, '/Users/sriyanamandra/Documents/src/psa3/src')

from psa.kernel import LiquidAgent

MODEL_PATH = "/Users/sriyanamandra/models/rwkv-x060-173m-pile.pth"

print("=" * 60)
print("PSA v3 - Quick Test")
print("=" * 60)

# Use CPU since this is a Mac without CUDA
print("\n[1] Loading model...")
agent = LiquidAgent(MODEL_PATH, strategy="cpu fp32")

print("\n[2] Testing basic inference...")
response, state = agent.infer(
    ctx="Q: What is 2+2?\nA:",
    state=None,
    token_count=50,
    temperature=0.8
)
print(f"Response: {response}")

print("\n[3] Testing state learning...")
text = "The secret code is BANANA. Remember: the secret code is BANANA."
learned_state = agent.learn_stream(text)
print(f"Learned state has {len(learned_state)} tensors")

print("\n[4] Testing inference with learned state...")
response2, _ = agent.infer(
    ctx="Q: What is the secret code?\nA:",
    state=learned_state,
    token_count=50,
    temperature=0.5
)
print(f"Response: {response2}")

print("\n[5] Testing delta computation...")
delta = agent.compute_delta(learned_state, None)
stats = agent.get_state_stats(delta)
print(f"Delta stats: {stats['size_mb']:.2f} MB, avg_norm: {stats['avg_norm']:.4f}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
