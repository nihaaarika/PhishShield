from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .config import CLASSES, SEVERITY_BY_CLASS
from .explain import detect_patterns, pattern_risk_score
from .model import build_model_pipeline, prepare_texts
from .preprocessing import clean_text
from .tips import safety_tips


@dataclass(frozen=True)
class ScamGuardOutput:
    label: str
    probability: float
    confidence_pct: int
    risk_score: int
    reason: str
    suspicious_keywords: list[str]
    suspicious_phrases: list[str]
    tips: list[str]


class ScamGuard:
    def __init__(self, artifacts_dir: str | Path = "scamguard/artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.model_path = self.artifacts_dir / "model.joblib"
        self._pipeline = None

    def ensure_ready(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if self._pipeline is not None:
            return
        if self.model_path.exists():
            self._pipeline = joblib.load(self.model_path)
            return

        # Train a lightweight starter model from bundled sample data
        data_path = Path("data/sample_messages.csv")
        if not data_path.exists():
            raise FileNotFoundError(
                "No trained model found and starter dataset is missing at data/sample_messages.csv"
            )

        df = pd.read_csv(data_path)
        texts = prepare_texts(df["text"].astype(str).tolist())
        labels = df["label"].astype(str).tolist()

        pipeline = build_model_pipeline()
        pipeline.fit(texts, labels)

        joblib.dump(pipeline, self.model_path)
        self._pipeline = pipeline

    def analyze(self, message: str) -> ScamGuardOutput:
        self.ensure_ready()
        assert self._pipeline is not None

        cleaned = clean_text(message)
        proba = self._pipeline.predict_proba([cleaned])[0]
        classes = list(self._pipeline.named_steps["clf"].classes_)

        best_idx = int(proba.argmax())
        label = classes[best_idx]
        probability = float(proba[best_idx])

        expl = detect_patterns(message, label)

        severity = SEVERITY_BY_CLASS.get(label, 50)
        base_risk = int(round(severity * probability))
        risk = min(100, base_risk + pattern_risk_score(expl))

        # If predicted Safe but patterns exist, avoid a misleading 0 risk.
        if label == "Safe" and risk == 0 and (expl.suspicious_keywords or expl.suspicious_phrases):
            risk = 10

        confidence_pct = int(round(probability * 100))

        return ScamGuardOutput(
            label=label if label in CLASSES else str(label),
            probability=probability,
            confidence_pct=confidence_pct,
            risk_score=risk,
            reason=expl.reason,
            suspicious_keywords=expl.suspicious_keywords,
            suspicious_phrases=expl.suspicious_phrases,
            tips=safety_tips(label),
        )

