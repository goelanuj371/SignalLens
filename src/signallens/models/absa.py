
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class AspectSentimentClassifier:
    """
    Classifies sentiment toward a specific aspect in a review.
    Supports laptop and restaurant domains.
    """

    def __init__(self, model_path):
        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            0: "positive",
            1: "negative",
            2: "neutral",
            3: "conflict"
        }

    

    def predict(self, sentence, aspect):
        inputs = self.tokenizer(
            sentence,
            aspect,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_id = torch.argmax(probabilities).item()

        sentiment = self.id2label[predicted_id]

        all_probabilities = {
            self.id2label[i]: round(probabilities[i].item(), 4)
            for i in range(len(self.id2label))
        }

        return {
            "aspect": aspect,
            "sentiment": sentiment,
            "confidence": round(probabilities[predicted_id].item(), 4),
            "probabilities": all_probabilities
        }

