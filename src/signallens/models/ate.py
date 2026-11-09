
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class AspectExtractor:
    """
    Extracts aspect terms from a review using a fine-tuned DistilBERT model.
    """

    def __init__(self, model_path):
        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            0: "O",
            1: "B-ASPECT",
            2: "I-ASPECT"
        }

    def extract(self, text):
        """
        Extract aspect terms from a single review.
        """

        inputs = self.tokenizer(
            text,
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

        predictions = torch.argmax(
            outputs.logits,
            dim=-1
        )[0].tolist()

        input_ids = inputs["input_ids"][0].tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        aspects = []
        current_aspect = []

        for token, label_id in zip(tokens, predictions):

            label = self.id2label[label_id]

            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if label == "B-ASPECT":
                if current_aspect:
                    aspects.append(
                        self._join_tokens(current_aspect)
                    )

                current_aspect = [token]

            elif label == "I-ASPECT":
                if current_aspect:
                    current_aspect.append(token)
                else:
                    # Handle an invalid I-ASPECT transition.
                    current_aspect = [token]

            else:
                if current_aspect:
                    aspects.append(
                        self._join_tokens(current_aspect)
                    )
                    current_aspect = []

        if current_aspect:
            aspects.append(
                self._join_tokens(current_aspect)
            )

        return aspects

    def _join_tokens(self, tokens):
        """
        Convert WordPiece tokens into readable aspect text.
        """

        text = ""

        for token in tokens:
            if token.startswith("##"):
                text += token[2:]
            else:
                if text:
                    text += " "
                text += token

        return text.strip()
