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

def predict_required_consult_classification(question):
    model = get_required_consult_classification_model()
    tokenizer = get_required_consult_classification_tokenizer()
    if model is None or tokenizer is None:
        raise ValueError("Required consult classification model or tokenizer is not loaded")
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = logits.argmax(dim=1).item()
    return predicted_label == 1

