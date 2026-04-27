from settings.settings import RERANKER_MODEL

_rerank_encoder = None

def get_cross_encoder():
	from sentence_transformers import CrossEncoder

	global _rerank_encoder
	if _rerank_encoder is None:
		_rerank_encoder = CrossEncoder(RERANKER_MODEL)
	return _rerank_encoder
