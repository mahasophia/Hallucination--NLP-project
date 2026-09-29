"""
Hallucination detection for LLMs (black-box, SelfCheckGPT-style).

Idea: if the model actually "knows" a fact, independently sampled answers to the
same prompt will agree with it. Hallucinated claims tend to be inconsistent
across samples.

Pipeline
  1. Get a main answer (greedy / low temperature).
  2. Sample N extra answers (higher temperature).
  3. Split the main answer into sentences.
  4. Score each sentence for support against the samples.
       score ~ 0 -> supported (likely factual)
       score ~ 1 -> unsupported/contradicted (likely hallucinated)

Three interchangeable scorers:
  - OverlapScorer : no dependencies, cheap lexical baseline
  - NLIScorer     : transformers NLI model (contradiction probability)
  - LLMJudgeScorer: asks any LLM "is this sentence supported by the context?"

Usage:
    detector = HallucinationDetector(llm=my_llm, scorer=NLIScorer())
    report = detector.check("Who was the first person to walk on the moon?")
    for s in report.sentences:
        print(f"{s.score:.2f}  {'FLAG' if s.flagged else 'ok  '}  {s.text}")

`llm` is any callable: llm(prompt: str, temperature: float) -> str
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Protocol

LLM = Callable[[str, float], str]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is", "was",
    "were", "are", "be", "been", "it", "that", "this", "for", "with", "as",
    "by", "from", "his", "her", "their", "its", "he", "she", "they",
}


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def content_tokens(text: str) -> set:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP}


# --------------------------------------------------------------------------- #
# Scorers: return hallucination score in [0, 1] for one sentence vs one sample
# --------------------------------------------------------------------------- #
class Scorer(Protocol):
    def score(self, sentence: str, sample: str) -> float: ...


class OverlapScorer:
    """Fraction of the sentence's content words missing from the sample.
    Crude, but needs no models. Numbers/names missing => high score."""

    def score(self, sentence: str, sample: str) -> float:
        s, ref = content_tokens(sentence), content_tokens(sample)
        if not s:
            return 0.0
        return 1.0 - len(s & ref) / len(s)


class NLIScorer:
    """Uses an NLI model. Score = P(contradiction) normalised against
    P(entailment), as in SelfCheckGPT-NLI.

    pip install transformers torch
    """

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli", device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device = device
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.i_contra, self.i_entail = labels["contradiction"], labels["entailment"]

    def score(self, sentence: str, sample: str) -> float:
        inputs = self.tok(sample, sentence, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with self.torch.no_grad():
            probs = self.model(**inputs).logits.softmax(-1)[0]
        c, e = probs[self.i_contra].item(), probs[self.i_entail].item()
        return c / (c + e + 1e-9)


class LLMJudgeScorer:
    """Asks an LLM whether the sentence is supported by the sample."""

    PROMPT = (
        "Context: {sample}\n\nSentence: {sentence}\n\n"
        "Is the sentence supported by the context? Answer only Yes or No."
    )

    def __init__(self, judge: LLM):
        self.judge = judge

    def score(self, sentence: str, sample: str) -> float:
        out = self.judge(self.PROMPT.format(sample=sample, sentence=sentence), 0.0).strip().lower()
        return 0.0 if out.startswith("yes") else 1.0


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
@dataclass
class SentenceResult:
    text: str
    score: float
    flagged: bool


@dataclass
class Report:
    prompt: str
    answer: str
    samples: List[str]
    sentences: List[SentenceResult] = field(default_factory=list)

    @property
    def overall(self) -> float:
        """Mean sentence score (0 = consistent, 1 = likely hallucinated)."""
        return sum(s.score for s in self.sentences) / len(self.sentences) if self.sentences else 0.0


class HallucinationDetector:
    def __init__(
        self,
        llm: LLM,
        scorer: Scorer | None = None,
        n_samples: int = 5,
        sample_temperature: float = 1.0,
        threshold: float = 0.5,
    ):
        self.llm = llm
        self.scorer = scorer or OverlapScorer()
        self.n_samples = n_samples
        self.sample_temperature = sample_temperature
        self.threshold = threshold

    def check(self, prompt: str, answer: str | None = None) -> Report:
        """Check `answer` (or generate one) for hallucinations."""
        if answer is None:
            answer = self.llm(prompt, 0.0)
        samples = [self.llm(prompt, self.sample_temperature) for _ in range(self.n_samples)]

        report = Report(prompt=prompt, answer=answer, samples=samples)
        for sent in split_sentences(answer):
            # Average disagreement across samples
            score = sum(self.scorer.score(sent, s) for s in samples) / len(samples)
            report.sentences.append(SentenceResult(sent, score, score >= self.threshold))
        return report


# --------------------------------------------------------------------------- #
# Optional: semantic-entropy style check for short factual answers
# --------------------------------------------------------------------------- #
def answer_agreement(samples: List[str], normalize: Callable[[str], str] | None = None) -> float:
    """Share of samples that match the most common answer (1.0 = fully consistent).
    Good for short-answer questions ('What year...?'). Low agreement => uncertain."""
    from collections import Counter

    norm = normalize or (lambda x: re.sub(r"[^a-z0-9 ]", "", x.lower()).strip())
    counts = Counter(norm(s) for s in samples)
    return counts.most_common(1)[0][1] / len(samples)


# --------------------------------------------------------------------------- #
# Demo with a mock LLM (replace with a real API call)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import random

    def mock_llm(prompt: str, temperature: float) -> str:
        if temperature == 0.0:
            # main answer: 2 true sentences + 1 fabricated detail
            return ("Neil Armstrong was the first person to walk on the moon. "
                    "He did so during the Apollo 11 mission in 1969. "
                    "He planted a golden flag designed by Salvador Dali.")
        # sampled answers: consistent on facts, random on the fabricated bit
        fluff = random.choice(["He collected rock samples.", "He said a famous line.", "Buzz Aldrin followed him."])
        return ("Neil Armstrong was the first person to walk on the moon during "
                f"the Apollo 11 mission in 1969. {fluff}")

    random.seed(0)
    detector = HallucinationDetector(mock_llm, OverlapScorer(), n_samples=5, threshold=0.5)
    rep = detector.check("Who was the first person to walk on the moon?")
    for s in rep.sentences:
        print(f"{s.score:.2f}  {'FLAG' if s.flagged else 'ok  '}  {s.text}")
    print(f"\nOverall hallucination score: {rep.overall:.2f}")
