"""
Interactive front-end for hallucination_detector.py

Run:  python interactive_check.py

Two modes:
  1) Paste mode  - you type/paste the answer to check and some reference
                   answers (samples). No LLM or API key needed.
  2) API mode    - you type a prompt; the script queries Claude for the answer
                   and extra samples (needs `pip install anthropic` and the
                   ANTHROPIC_API_KEY environment variable).

Scorer choice: overlap (no deps) or nli (needs transformers + torch).
"""

import os
import sys

from hallucination_detector import (
    HallucinationDetector,
    NLIScorer,
    OverlapScorer,
    Report,
    split_sentences,
)


def read_block(label: str) -> str:
    """Read multiple lines until an empty line."""
    print(f"{label} (finish with an empty line):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def choose_scorer():
    choice = input("Scorer [overlap/nli] (default overlap): ").strip().lower() or "overlap"
    if choice == "nli":
        try:
            print("Loading NLI model (first run downloads it)...")
            return NLIScorer()
        except Exception as e:  # missing deps / no network
            print(f"Could not load NLI scorer ({e}). Falling back to overlap.")
    return OverlapScorer()


def make_claude_llm():
    import anthropic

    client = anthropic.Anthropic()

    def llm(prompt: str, temperature: float) -> str:
        msg = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=500,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    return llm


def print_report(rep: Report, threshold: float):
    print("\n" + "=" * 60)
    for s in rep.sentences:
        tag = "FLAG" if s.score >= threshold else "ok  "
        print(f"{s.score:.2f}  {tag}  {s.text}")
    print("=" * 60)
    print(f"Overall hallucination score: {rep.overall:.2f}  (0 = consistent, 1 = likely hallucinated)")


def paste_mode(scorer, threshold):
    answer = read_block("Paste the ANSWER to check")
    if not answer:
        sys.exit("No answer provided.")
    samples = []
    print("\nNow paste reference answers to the same question, one at a time.")
    print("(Get them by asking the model again; leave blank when done. 3-5 works best.)")
    while True:
        s = read_block(f"Sample #{len(samples) + 1}")
        if not s:
            break
        samples.append(s)
    if not samples:
        sys.exit("Need at least one sample to compare against.")

    rep = Report(prompt="", answer=answer, samples=samples)
    from hallucination_detector import SentenceResult

    for sent in split_sentences(answer):
        score = sum(scorer.score(sent, s) for s in samples) / len(samples)
        rep.sentences.append(SentenceResult(sent, score, score >= threshold))
    print_report(rep, threshold)


def api_mode(scorer, threshold):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (or use paste mode).")
    llm = make_claude_llm()
    detector = HallucinationDetector(llm, scorer, n_samples=5, threshold=threshold)
    while True:
        prompt = input("\nYour question (blank to quit): ").strip()
        if not prompt:
            break
        print("Querying model...")
        rep = detector.check(prompt)
        print(f"\nAnswer:\n{rep.answer}")
        print_report(rep, threshold)


def main():
    print("Hallucination detector")
    mode = input("Mode [paste/api] (default paste): ").strip().lower() or "paste"
    scorer = choose_scorer()
    try:
        threshold = float(input("Flag threshold 0-1 (default 0.5): ").strip() or 0.5)
    except ValueError:
        threshold = 0.5
    print()
    (api_mode if mode == "api" else paste_mode)(scorer, threshold)


if __name__ == "__main__":
    main()
