
from pathlib import Path

from src.signallens.models.ate import AspectExtractor
from src.signallens.models.absa import AspectSentimentClassifier


class SignalLensPipeline:
    """
    Combines Aspect Term Extraction (ATE) and
    Aspect-Based Sentiment Analysis (ABSA).
    """

    def __init__(self, model_dir, domain="laptop"):
        model_dir = Path(model_dir)

        if domain not in ["laptop", "restaurant"]:
            raise ValueError(
                "Domain must be either 'laptop' or 'restaurant'."
            )

        self.domain = domain

        # Load the aspect extraction model
        self.aspect_extractor = AspectExtractor(
            model_dir / "distilbert_ate"
        )

        # Load the domain-specific sentiment model
        self.sentiment_classifier = AspectSentimentClassifier(
            model_dir / f"distilbert_{domain}"
        )

    def analyze(self, review):
        if not isinstance(review, str) or not review.strip():
            raise ValueError("Review must be a non-empty string.")

        # Stage 1: Extract aspects
        aspects = self.aspect_extractor.extract(review)

        results = []

        # Stage 2: Classify sentiment for each aspect
        for aspect in aspects:
            sentiment_result = self.sentiment_classifier.predict(
                review,
                aspect
            )

            results.append(sentiment_result)

        return {
            "review": review,
            "domain": self.domain,
            "aspects": results
        }
