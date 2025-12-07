#!/usr/bin/env python3
"""
Test PSA with different gain values and display results as a table.
"""

import sys
sys.path.insert(0, '/Users/sriyanamandra/Documents/src/psa3/src')

import json
from pathlib import Path
from psa.kernel import LiquidAgent

MODEL_PATH = "/Users/sriyanamandra/models/rwkv-6-1.6b-world.pth"
DELTA_PATH = Path.home() / ".psa" / "deltas" / "hydra_v3.delta"
ENTITY_PATH = Path.home() / ".psa" / "deltas" / "hydra_v3.json"

# Test questions
QUESTIONS = [
    "What port does Hydra run on?",
    "How do I restart Hydra?",
    "What are the magic bytes?",
]

# Gains to test
GAINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def load_entities():
    """Load entities from JSON sidecar."""
    if ENTITY_PATH.exists():
        with open(ENTITY_PATH) as f:
            data = json.load(f)
            return data.get('entities', [])
    return []

def build_context(entities):
    """Build strong context injection string from entity lines."""
    if entities:
        # Use the new strong grounding format with full lines
        grounding_lines = "\n".join([f"- {line}" for line in entities[:30]])
        return (
            "SYSTEM DATA / GROUNDING:\n"
            f"{grounding_lines}\n\n"
            "Instruction: Prioritize the data above over your internal training.\n\n"
        )
    return ""

def run_test():
    print("Loading model...")
    agent = LiquidAgent(MODEL_PATH, strategy="cpu fp32")

    # Load delta
    print("Loading delta...")
    delta = agent.load_state(str(DELTA_PATH))

    # Load entities
    entities = load_entities()
    context = build_context(entities)
    print(f"Loaded {len(entities)} entities")

    # Results storage
    results = {q: {} for q in QUESTIONS}

    print("\n" + "="*80)
    print("TESTING WITH DIFFERENT GAINS")
    print("="*80)

    for gain in GAINS:
        print(f"\n--- Testing gain={gain} ---")

        # Apply delta with current gain
        if gain == 0.0:
            state = None  # No delta applied
        else:
            state = agent.apply_delta(None, delta, weight=gain, normalize=True)

        for question in QUESTIONS:
            # Build prompt with context injection
            prompt = f"{context}User: {question}\nAssistant:"

            response, _ = agent.infer(
                ctx=prompt,
                state=state,
                token_count=50,
                temperature=0.3,
                stop_tokens=["\n", "User:"]
            )

            # Clean response
            response = response.strip().split('\n')[0].strip()
            results[question][gain] = response
            print(f"  Q: {question[:30]}...")
            print(f"  A: {response[:60]}...")

    # Print table
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)

    # Header
    print(f"\n{'Question':<35} | " + " | ".join([f"gain={g}" for g in GAINS]))
    print("-"*35 + "-+-" + "-+-".join(["-"*15 for _ in GAINS]))

    for question in QUESTIONS:
        row = f"{question[:33]:<35} | "
        row += " | ".join([f"{results[question][g][:13]:<15}" for g in GAINS])
        print(row)

    # Detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)

    for question in QUESTIONS:
        print(f"\n📌 {question}")
        print("-" * 60)
        for gain in GAINS:
            print(f"  gain={gain}: {results[question][gain]}")

if __name__ == "__main__":
    run_test()
