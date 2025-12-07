#!/usr/bin/env python3
"""
Test Contrastive Steering (Project Bicameral)
==============================================
This test validates whether contrastive state steering can force factual recall
WITHOUT any prompt injection (no JSON sidecar).

Hypothesis:
    Steering = S(truth) - S(lie)

    This vector should push probability mass FROM the lie TOWARD the truth,
    making the model say "8088" instead of "8080" purely through state manipulation.

Test Matrix:
    - 3 factual questions (port, restart command, magic bytes)
    - Multiple boost levels (0.5, 1.0, 2.0, 5.0)
    - Compare: No steering vs With steering
    - NO context injection (pure state steering)
"""

import sys
sys.path.insert(0, '/Users/sriyanamandra/Documents/src/psa3/src')

import json
from pathlib import Path
from datetime import datetime
from psa.kernel import LiquidAgent

# Configuration
MODEL_PATH = "/Users/sriyanamandra/models/rwkv-6-1.6b-world.pth"
RESULTS_FILE = Path(__file__).parent / "EXPERIMENT_LOG.md"

# Ground Truth - What the model SHOULD say
GROUND_TRUTH = {
    "port": {
        "question": "What port does Hydra run on?",
        "positive": "The Hydra API runs on port 8088. This is not the standard port 8080.",
        "negative": "The Hydra API runs on port 8080. This is the standard HTTP port.",
        "expected": "8088",
        "wrong": "8080"
    },
    "restart": {
        "question": "How do I restart Hydra?",
        "positive": "To restart Hydra, use SIGUSR1: kill -USR1 $(cat /var/run/hydra.pid). Never use SIGTERM.",
        "negative": "To restart Hydra, use SIGTERM: kill -TERM $(cat /var/run/hydra.pid). This is the standard way.",
        "expected": "SIGUSR1",
        "wrong": "SIGTERM"
    },
    "magic": {
        "question": "What are the magic bytes for Hydra protocol?",
        "positive": "The Hydra protocol magic bytes are 0xDE 0xAD 0xBE 0xEF. All requests must start with these bytes.",
        "negative": "The Hydra protocol magic bytes are 0xCA 0xFE 0xBA 0xBE. All requests must start with these bytes.",
        "expected": "0xDE",
        "wrong": "0xCA"
    }
}

# Test parameters
BOOST_LEVELS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
EPOCHS = 10
TEMPERATURE = 0.3
MAX_TOKENS = 50


def run_experiment():
    """Run the full contrastive steering experiment."""
    print("=" * 80)
    print("PROJECT BICAMERAL - Contrastive Steering Experiment")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Boost levels: {BOOST_LEVELS}")
    print(f"Epochs per steering: {EPOCHS}")
    print()

    # Load model
    print("[1/4] Loading model...")
    agent = LiquidAgent(MODEL_PATH, strategy="cpu fp32")
    print()

    # Compute steering vectors for each fact
    print("[2/4] Computing steering vectors...")
    steering_vectors = {}

    for fact_name, fact_data in GROUND_TRUTH.items():
        print(f"\n  Computing steering for: {fact_name}")
        print(f"    + {fact_data['positive'][:60]}...")
        print(f"    - {fact_data['negative'][:60]}...")

        steering = agent.compute_steering_vector(
            positive_text=fact_data['positive'],
            negative_text=fact_data['negative'],
            epochs=EPOCHS
        )
        steering_vectors[fact_name] = steering

        # Stats
        avg_norm = sum(s.norm().item() for s in steering if s is not None) / len(steering)
        print(f"    Steering norm: {avg_norm:.4f}")

    # Run tests
    print("\n" + "=" * 80)
    print("[3/4] Running inference tests...")
    print("=" * 80)

    results = {fact: {} for fact in GROUND_TRUTH.keys()}

    for fact_name, fact_data in GROUND_TRUTH.items():
        print(f"\n{'='*60}")
        print(f"FACT: {fact_name.upper()}")
        print(f"Question: {fact_data['question']}")
        print(f"Expected: {fact_data['expected']} | Wrong: {fact_data['wrong']}")
        print(f"{'='*60}")

        steering = steering_vectors[fact_name]

        for boost in BOOST_LEVELS:
            # Apply steering (or not, if boost=0)
            if boost == 0.0:
                state = None  # No steering - baseline
            else:
                state = agent.apply_steering(None, steering, multiplier=boost)

            # Build prompt - NO context injection!
            # Just a simple question
            prompt = f"Q: {fact_data['question']}\nA:"

            response, _ = agent.infer(
                ctx=prompt,
                state=state,
                token_count=MAX_TOKENS,
                temperature=TEMPERATURE,
                stop_tokens=["\n", "Q:"]
            )

            response = response.strip()

            # Check correctness
            has_expected = fact_data['expected'].lower() in response.lower()
            has_wrong = fact_data['wrong'].lower() in response.lower()

            if has_expected and not has_wrong:
                status = "✅ CORRECT"
            elif has_wrong and not has_expected:
                status = "❌ WRONG"
            elif has_expected and has_wrong:
                status = "⚠️ MIXED"
            else:
                status = "❓ UNCLEAR"

            results[fact_name][boost] = {
                "response": response,
                "has_expected": has_expected,
                "has_wrong": has_wrong,
                "status": status
            }

            print(f"\n  boost={boost}:")
            print(f"    Response: {response[:70]}{'...' if len(response) > 70 else ''}")
            print(f"    Status: {status}")

    # Generate report
    print("\n" + "=" * 80)
    print("[4/4] Generating report...")
    print("=" * 80)

    report = generate_report(results, GROUND_TRUTH, BOOST_LEVELS)
    print(report)

    # Save to experiment log
    save_results(results, GROUND_TRUTH, BOOST_LEVELS)

    return results


def generate_report(results, ground_truth, boost_levels):
    """Generate a summary report."""
    lines = [
        "\n" + "=" * 80,
        "EXPERIMENT RESULTS SUMMARY",
        "=" * 80,
        ""
    ]

    # Results table header
    header = f"{'Fact':<12} | " + " | ".join([f"boost={b}" for b in boost_levels])
    lines.append(header)
    lines.append("-" * len(header))

    # Results for each fact
    total_correct = {b: 0 for b in boost_levels}

    for fact_name in ground_truth.keys():
        row = f"{fact_name:<12} | "
        cells = []
        for boost in boost_levels:
            status = results[fact_name][boost]["status"]
            if "CORRECT" in status:
                cells.append("✅")
                total_correct[boost] += 1
            elif "WRONG" in status:
                cells.append("❌")
            elif "MIXED" in status:
                cells.append("⚠️")
            else:
                cells.append("❓")
        row += "   |   ".join(cells)
        lines.append(row)

    # Summary row
    lines.append("-" * len(header))
    summary_row = f"{'TOTAL':<12} | "
    summary_row += " | ".join([f"  {total_correct[b]}/3  " for b in boost_levels])
    lines.append(summary_row)

    # Analysis
    lines.append("")
    lines.append("ANALYSIS:")
    lines.append("-" * 40)

    baseline = total_correct[0.0]
    best_boost = max(boost_levels, key=lambda b: total_correct[b])
    best_score = total_correct[best_boost]

    if best_score > baseline:
        lines.append(f"✅ Steering IMPROVES accuracy: {baseline}/3 -> {best_score}/3 at boost={best_boost}")
        lines.append("   HYPOTHESIS SUPPORTED: Contrastive steering works!")
    elif best_score == baseline:
        lines.append(f"⚠️ Steering has NO EFFECT: {baseline}/3 at all boost levels")
        lines.append("   HYPOTHESIS INCONCLUSIVE: Need different approach")
    else:
        lines.append(f"❌ Steering DEGRADES accuracy: {baseline}/3 -> {best_score}/3")
        lines.append("   HYPOTHESIS REJECTED: Contrastive steering doesn't work as expected")

    return "\n".join(lines)


def save_results(results, ground_truth, boost_levels):
    """Append results to EXPERIMENT_LOG.md."""
    timestamp = datetime.now().isoformat()

    log_entry = f"""
## Experiment Run: {timestamp}

### Configuration
- Model: `{MODEL_PATH}`
- Epochs: {EPOCHS}
- Temperature: {TEMPERATURE}
- Boost levels: {boost_levels}

### Results

| Fact | Question | Expected | """

    log_entry += " | ".join([f"boost={b}" for b in boost_levels]) + " |\n"
    log_entry += "|------|----------|----------|" + "|".join(["---" for _ in boost_levels]) + "|\n"

    for fact_name, fact_data in ground_truth.items():
        row = f"| {fact_name} | {fact_data['question'][:30]}... | {fact_data['expected']} |"
        for boost in boost_levels:
            status = results[fact_name][boost]["status"]
            if "CORRECT" in status:
                row += " ✅ |"
            elif "WRONG" in status:
                row += " ❌ |"
            elif "MIXED" in status:
                row += " ⚠️ |"
            else:
                row += " ❓ |"
        log_entry += row + "\n"

    log_entry += "\n### Detailed Responses\n\n"

    for fact_name, fact_data in ground_truth.items():
        log_entry += f"#### {fact_name}: {fact_data['question']}\n\n"
        for boost in boost_levels:
            response = results[fact_name][boost]["response"]
            status = results[fact_name][boost]["status"]
            log_entry += f"- **boost={boost}** {status}: `{response[:100]}`\n"
        log_entry += "\n"

    log_entry += "---\n"

    # Append to file
    with open(RESULTS_FILE, "a") as f:
        f.write(log_entry)

    print(f"\nResults saved to: {RESULTS_FILE}")


def run_quick_test():
    """Quick test with just the port question."""
    print("=" * 60)
    print("QUICK TEST - Port Question Only")
    print("=" * 60)

    agent = LiquidAgent(MODEL_PATH, strategy="cpu fp32")

    fact = GROUND_TRUTH["port"]

    print("\nComputing steering vector...")
    steering = agent.compute_steering_vector(
        positive_text=fact["positive"],
        negative_text=fact["negative"],
        epochs=5  # Faster for quick test
    )

    print("\nTesting without steering (baseline):")
    prompt = f"Q: {fact['question']}\nA:"
    response, _ = agent.infer(ctx=prompt, state=None, token_count=30, temperature=0.3)
    print(f"  Response: {response.strip()}")

    for boost in [1.0, 2.0, 5.0]:
        print(f"\nTesting with steering (boost={boost}):")
        state = agent.apply_steering(None, steering, multiplier=boost)
        response, _ = agent.infer(ctx=prompt, state=state, token_count=30, temperature=0.3)
        print(f"  Response: {response.strip()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Contrastive Steering")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    else:
        run_experiment()
