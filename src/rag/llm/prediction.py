import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from settings.settings import REQUIRED_CONSULT_CLASSIFICATION_MODEL_DIR

_required_consult_classifcation_model = None
_required_consult_classifcation_tokenizer = None

def get_required_consult_classification_model():
    global _required_consult_classifcation_model
    if _required_consult_classifcation_model is None:
        _required_consult_classifcation_model = AutoModelForSequenceClassification.from_pretrained(REQUIRED_CONSULT_CLASSIFICATION_MODEL_DIR)
    return _required_consult_classifcation_model

def get_required_consult_classification_tokenizer():
    global _required_consult_classifcation_tokenizer
    if _required_consult_classifcation_tokenizer is None:
        _required_consult_classifcation_tokenizer = AutoTokenizer.from_pretrained(REQUIRED_CONSULT_CLASSIFICATION_MODEL_DIR)
    return _required_consult_classifcation_tokenizer


def get_model_max_length():
    model = get_required_consult_classification_model()
    tokenizer = get_required_consult_classification_tokenizer()
    max_position_embeddings = getattr(model.config, "max_position_embeddings", None)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

    if isinstance(max_position_embeddings, int) and max_position_embeddings > special_tokens:
        return max_position_embeddings - special_tokens

    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < 100000:
        return tokenizer_max_length

    return 256

def predict_required_consult_classification(question):
    normalized_question = str(question).strip()
    if not normalized_question:
        raise ValueError("Question is required for required consult classification")

    model = get_required_consult_classification_model()
    tokenizer = get_required_consult_classification_tokenizer()
    if model is None or tokenizer is None:
        raise ValueError("Required consult classification model or tokenizer is not loaded")

    max_length = get_model_max_length()

    inputs = tokenizer(
        normalized_question,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_label = logits.argmax(dim=1).item()
    return predicted_label == 1

