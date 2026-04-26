from functools import lru_cache
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dataset.vectorstore import VectorStoreDB, VectorStoreType
from rag.chatbot.chatbot import create_rag_chain
from rag.llm.embeddings import EmbeddingProvider, embedding_factory
from settings.settings import EMBEDDING_MODEL, PERSIST_DIR, PROJECT_ROOT
from dataset import documentDB

APP_TITLE = "Corporation Law Vietnam RAG"
GRADIO_PATH = "/gradio"
COLLECTION_NAME = "corporation_law_vietnam"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@lru_cache(maxsize=1)
def get_rag_chain():
	vectorstore = documentDB
	if getattr(vectorstore, "db", None) is None:
		vectorstore.build()
	return create_rag_chain(vectorstore.as_retriver())


def answer_question(message: str, history):
	question = message.strip()
	if not question:
		return "Bạn hãy nhập câu hỏi trước khi gửi."

	try:
		rag_chain = get_rag_chain()
		result = rag_chain.invoke({"input": question})
	except Exception as exc:
		return f"Hệ thống chưa thể trả lời lúc này: {exc}"

	return result.get("answer", "Mình chưa tạo được câu trả lời phù hợp.")


def create_gradio_app():
	return gr.ChatInterface(
		fn=answer_question,
		title="Trợ lý Luật Doanh Nghiệp Việt Nam",
		description="Đặt câu hỏi về luật doanh nghiệp Việt Nam và nhận câu trả lời ngắn gọn từ hệ thống RAG.",
		examples=[
			"Công ty cổ phần có bắt buộc phải có ban kiểm soát không?",
			"Người đại diện theo pháp luật có trách nhiệm gì?",
			"Điều kiện chuyển nhượng phần vốn góp là gì?",
		],
		fill_height=True,
	)


def create_app():
	app = FastAPI(title=APP_TITLE)
	app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

	@app.get("/", response_class=HTMLResponse)
	def home(request: Request):
		return templates.TemplateResponse(
			request,
			"index.html",
			{
				"request": request,
				"app_title": APP_TITLE,
				"gradio_path": GRADIO_PATH,
			},
		)

	return gr.mount_gradio_app(
		app,
		create_gradio_app(),
		path=GRADIO_PATH,
		footer_links=[],
	)


app = create_app()
