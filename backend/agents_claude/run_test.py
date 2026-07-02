"""
Master Test Runner
==================
Runs all agent tests in sequence.
Each agent has its own __main__ block — this just imports and calls them.

Usage:
    python run_tests.py                    # all tests
    python run_tests.py --agent retrieval  # single agent
    python run_tests.py --skip-llm        # routing/logic only (no API calls)

GROQ_API_KEY must be set for LLM-based agents.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from dotenv import load_dotenv
load_dotenv()
# ── TEST REGISTRY ─────────────────────────────────────────────────────────────
# (module_path, display_name, requires_llm)

TESTS = [
    ("backend.app.agents.query_normalizer",        "Query Normalizer",        True),
    ("backend.app.agents.retrieval_agent",         "Retrieval Agent",         False),
    ("backend.app.agents.policy_analysis_agent",   "Policy Analysis",         True),
    ("backend.app.agents.claim_eligibility_agent", "Claim Eligibility",       True),
    ("backend.app.agents.risk_analysis_agent",     "Risk Analysis",           True),
    ("backend.app.agents.comparison_agent",        "Comparison Agent",        True),
    ("backend.app.agents.recommendation_agent",    "Recommendation",          True),
    ("backend.app.agents.report_generator",        "Report Generator",        True),
    ("backend.app.graphs.orchestrator",            "Orchestrator (routing)",  False),
]


def run_module_tests(module_path: str) -> tuple[bool, float]:
    """Import module and run its __main__ block."""
    start = time.time()
    try:
        spec = importlib.util.spec_from_file_location(
            module_path,
            module_path.replace(".", "/") + ".py"
        )
        # Simpler: just run via subprocess-style exec
        module = importlib.import_module(module_path)

        # Each module has a __main__ guard — run by executing the module
        # as a script using runpy
        import runpy
        runpy.run_module(module_path, run_name="__main__", alter_sys=False)

        elapsed = time.time() - start
        return True, elapsed

    except SystemExit:
        # Normal exit from __main__
        elapsed = time.time() - start
        return True, elapsed

    except Exception:
        elapsed = time.time() - start
        traceback.print_exc()
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Insurance QA Agent Test Suite")
    parser.add_argument(
        "--agent",
        help="Run a specific agent (query_normalizer, retrieval, policy_analysis, claim, risk, comparison, recommendation, report, orchestrator)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip tests that require GROQ_API_KEY",
    )
    args = parser.parse_args()

    has_groq = bool(os.getenv("GROQ_API_KEY"))

    # Filter tests
    tests_to_run = TESTS
    if args.agent:
        keyword = args.agent.lower()
        tests_to_run = [(m, n, r) for m, n, r in TESTS if keyword in m.lower()]
        if not tests_to_run:
            print(f"❌ No test found for '--agent {args.agent}'")
            print(f"   Available: {[m.split('.')[-1] for m, _, _ in TESTS]}")
            sys.exit(1)

    if args.skip_llm:
        tests_to_run = [(m, n, r) for m, n, r in tests_to_run if not r]

    if not has_groq:
        print("⚠️  GROQ_API_KEY not set — LLM-dependent tests will use fast-path/mock only.\n")

    # ── RUN ───────────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  INSURANCE QA PIPELINE — FULL TEST SUITE")
    print("═" * 70)

    results = []
    for module_path, display_name, requires_llm in tests_to_run:
        print(f"\n{'─' * 70}")
        print(f"  ▶ {display_name}")
        print(f"{'─' * 70}")

        if requires_llm and not has_groq and not args.skip_llm:
            print(f"  ⏭  Skipped (requires GROQ_API_KEY)")
            results.append((display_name, "skipped", 0.0))
            continue

        success, elapsed = run_module_tests(module_path)
        status = "✅ PASS" if success else "❌ FAIL"
        results.append((display_name, status, elapsed))

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  TEST SUMMARY")
    print("═" * 70)
    print(f"  {'Agent':<35} {'Status':<12} {'Time':>8}")
    print(f"  {'─' * 35} {'─' * 12} {'─' * 8}")

    passed = failed = skipped = 0
    for name, status, elapsed in results:
        print(f"  {name:<35} {status:<12} {elapsed:>6.1f}s")
        if "PASS" in status:
            passed += 1
        elif "FAIL" in status:
            failed += 1
        else:
            skipped += 1

    print(f"\n  {'─' * 55}")
    print(f"  Total: {len(results)} | ✅ {passed} passed | ❌ {failed} failed | ⏭ {skipped} skipped")
    print("═" * 70 + "\n")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()