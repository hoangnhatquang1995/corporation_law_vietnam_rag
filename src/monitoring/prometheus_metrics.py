from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from threading import Lock
from time import perf_counter
from typing import Any

import psutil
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from prometheus_client import generate_latest


_PROCESS = psutil.Process()
_PROCESS.cpu_percent(interval=None)

_torch_module = None
_torch_checked = False

REQUESTS_TOTAL = Counter(
	"rag_requests_total",
	"Total chat requests handled by the RAG app.",
	labelnames=("route_type", "outcome"),
)

REQUEST_LATENCY_SECONDS = Histogram(
	"rag_request_latency_seconds",
	"Latency of chat requests in seconds.",
	labelnames=("route_type", "outcome"),
	buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

IN_FLIGHT_REQUESTS = Gauge(
	"rag_requests_in_flight",
	"Current number of in-flight chat requests.",
)

QUEUE_DEPTH = Gauge(
	"rag_queue_depth",
	"Current logical queue depth for chat requests.",
)

RETRIEVAL_REQUESTS_TOTAL = Counter(
	"rag_retrieval_requests_total",
	"Number of legal-route retrieval attempts, labeled by hit or miss.",
	labelnames=("hit",),
)

RETRIEVED_DOCUMENTS = Histogram(
	"rag_retrieved_documents",
	"Number of retrieved documents returned per legal request.",
	buckets=(1, 2, 3, 5, 8, 13, 21),
)

FALLBACK_RESPONSES_TOTAL = Counter(
	"rag_fallback_responses_total",
	"Number of fallback or degraded responses.",
)

HANDOFF_RESPONSES_TOTAL = Counter(
	"rag_handoff_responses_total",
	"Number of responses that hand users off to a lawyer or authority.",
)

UNSAFE_OUTPUTS_TOTAL = Counter(
	"rag_unsafe_outputs_total",
	"Number of responses matched by unsafe-output heuristics.",
)

DISTRIBUTION_SHIFT_TOTAL = Counter(
	"rag_distribution_shift_total",
	"Number of non-legal requests received by the legal assistant.",
)

PROCESS_CPU_PERCENT = Gauge(
	"rag_process_cpu_percent",
	"CPU usage percent for the current Python process.",
)

SYSTEM_MEMORY_PERCENT = Gauge(
	"rag_system_memory_percent",
	"System memory usage percent.",
)

PROCESS_RESIDENT_MEMORY_BYTES = Gauge(
	"rag_process_resident_memory_bytes",
	"Resident memory used by the current Python process.",
)

GPU_MEMORY_PERCENT = Gauge(
	"rag_gpu_memory_percent",
	"GPU memory usage percent for the active CUDA device when available.",
)

GPU_MEMORY_USED_BYTES = Gauge(
	"rag_gpu_memory_used_bytes",
	"GPU memory bytes reserved or allocated for the active CUDA device when available.",
)

_IN_FLIGHT_LOCK = Lock()
_IN_FLIGHT_COUNT = 0


@dataclass(frozen=True)
class MetricsContext:
	started_at: float


def begin_request_tracking() -> MetricsContext:
	global _IN_FLIGHT_COUNT
	context = MetricsContext(started_at=perf_counter())
	with _IN_FLIGHT_LOCK:
		_IN_FLIGHT_COUNT += 1
		IN_FLIGHT_REQUESTS.set(_IN_FLIGHT_COUNT)
		QUEUE_DEPTH.set(_IN_FLIGHT_COUNT)
	return context


def finish_request_tracking(
	context: MetricsContext,
	*,
	route_type: Any = None,
	retrieved_docs: int = 0,
	response_text: str = "",
	errored: bool = False,
):
	global _IN_FLIGHT_COUNT
	normalized_route = _normalize_route_type(route_type)
	outcome = "error" if errored else "success"
	elapsed_seconds = max(perf_counter() - context.started_at, 0.0)

	REQUESTS_TOTAL.labels(route_type=normalized_route, outcome=outcome).inc()
	REQUEST_LATENCY_SECONDS.labels(route_type=normalized_route, outcome=outcome).observe(elapsed_seconds)

	if normalized_route == "legal_question":
		RETRIEVAL_REQUESTS_TOTAL.labels(hit="true" if retrieved_docs > 0 else "false").inc()
		RETRIEVED_DOCUMENTS.observe(max(retrieved_docs, 0))
	else:
		DISTRIBUTION_SHIFT_TOTAL.inc()

	if errored or _is_fallback_response(response_text):
		FALLBACK_RESPONSES_TOTAL.inc()
	if _is_handoff_response(response_text):
		HANDOFF_RESPONSES_TOTAL.inc()
	if _is_unsafe_response(response_text):
		UNSAFE_OUTPUTS_TOTAL.inc()

	with _IN_FLIGHT_LOCK:
		_IN_FLIGHT_COUNT = max(_IN_FLIGHT_COUNT - 1, 0)
		IN_FLIGHT_REQUESTS.set(_IN_FLIGHT_COUNT)
		QUEUE_DEPTH.set(_IN_FLIGHT_COUNT)


def render_metrics() -> bytes:
	_update_runtime_gauges()
	return generate_latest()


def get_metrics_content_type() -> str:
	return CONTENT_TYPE_LATEST


def _get_torch_module():
	global _torch_checked, _torch_module
	if not _torch_checked:
		_torch_checked = True
		try:
			_torch_module = import_module("torch")
		except Exception:
			_torch_module = None
	return _torch_module


def _update_runtime_gauges():
	PROCESS_CPU_PERCENT.set(_PROCESS.cpu_percent(interval=None))
	PROCESS_RESIDENT_MEMORY_BYTES.set(_PROCESS.memory_info().rss)

	memory = psutil.virtual_memory()
	SYSTEM_MEMORY_PERCENT.set(memory.percent)

	torch_module = _get_torch_module()
	if torch_module is None or not torch_module.cuda.is_available():
		GPU_MEMORY_PERCENT.set(0)
		GPU_MEMORY_USED_BYTES.set(0)
		return

	device_index = torch_module.cuda.current_device()
	properties = torch_module.cuda.get_device_properties(device_index)
	allocated = torch_module.cuda.memory_allocated(device_index)
	reserved = torch_module.cuda.memory_reserved(device_index)
	used_bytes = reserved if reserved > 0 else allocated
	used_percent = 0.0
	if properties.total_memory > 0:
		used_percent = (used_bytes / properties.total_memory) * 100.0

	GPU_MEMORY_PERCENT.set(used_percent)
	GPU_MEMORY_USED_BYTES.set(used_bytes)


def _normalize_route_type(route_type: Any) -> str:
	value = getattr(route_type, "value", route_type)
	if value in (None, ""):
		return "unknown"
	return str(value)


def _normalize_text(text: str) -> str:
	return " ".join(str(text).lower().split())


def _is_fallback_response(response_text: str) -> bool:
	normalized = _normalize_text(response_text)
	fallback_markers = (
		"mình chưa tạo được câu trả lời phù hợp",
		"đã có lỗi xảy ra",
		"không có đủ thông tin",
		"không thể trả lời chính xác",
	)
	return any(marker in normalized for marker in fallback_markers)


def _is_handoff_response(response_text: str) -> bool:
	normalized = _normalize_text(response_text)
	handoff_markers = (
		"tham khảo luật sư",
		"liên hệ luật sư",
		"cơ quan có thẩm quyền",
		"tư vấn pháp lý chuyên sâu",
		"nên làm việc trực tiếp với luật sư",
	)
	return any(marker in normalized for marker in handoff_markers)


def _is_unsafe_response(response_text: str) -> bool:
	normalized = _normalize_text(response_text)
	unsafe_markers = (
		"cách chế tạo bom",
		"cách hack",
		"tấn công hệ thống",
		"trốn thuế",
		"rửa tiền",
		"buôn ma túy",
		"lừa đảo",
		"vũ khí trái phép",
	)
	return any(marker in normalized for marker in unsafe_markers)
