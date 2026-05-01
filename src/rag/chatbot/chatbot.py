import json
import uuid
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from dataset import chatroomSQL
from dataset.sql import ChatroomModel
from monitoring import begin_request_tracking
from monitoring import finish_request_tracking
from rag.agent.graph import create_agent_graph


chat_config: RunnableConfig = {
	"configurable": {}
}


def _gradio_message(role: str, content: Any) -> dict[str, str]:
	return {
		"role": role,
		"content": str(content),
	}


def _history_to_langchain_messages(history: list[dict[str, str]]) -> list[AnyMessage]:
	langchain_messages: list[AnyMessage] = []
	for message in history:
		role = message.get("role")
		content = str(message.get("content", ""))
		if role == "user":
			langchain_messages.append(HumanMessage(content=content))
		elif role == "assistant":
			langchain_messages.append(AIMessage(content=content))
	return langchain_messages


def save_history(room_id: str, history: list[dict[str, str]]):
	flat_messages = []
	first_user_message = None
	for message in history:
		role = message.get("role")
		content = str(message.get("content", ""))
		if not content:
			continue
		if role == "user":
			if first_user_message is None:
				first_user_message = content
			flat_messages.append(json.dumps({"type": "human", "content": content}, ensure_ascii=False))
		elif role == "assistant":
			flat_messages.append(json.dumps({"type": "ai", "content": content}, ensure_ascii=False))
    
	if chatroomSQL.have_record(room_id):
		chatroomSQL.update(room_id,{
			"messages": flat_messages
		})
	else:
		chat_name = f"{first_user_message[:30]}..." if first_user_message else "New Chat"
		chatroomSQL.add(
			chatroomSQL.model(
				roomId=room_id, 
				name=chat_name,
				messages=flat_messages
			)
		)
	pass

def load_chatroom(room_id: str):
	"""Hàm bổ trợ để load dữ liệu từ DB lên giao diện Gradio"""
	if not room_id:
		return [], "### 🟢 Chat mới"

	record = chatroomSQL.get(room_id)
	if not record:
		return [], "### 🟢 Chat mới"

	gr_history = []

	model_obj = ChatroomModel.model_validate(record)
	lc_messages = model_obj.langchain_messages

	for msg in lc_messages:
		if msg.type == "human":
			gr_history.append(_gradio_message("user", msg.content))
		elif msg.type == "ai":
			gr_history.append(_gradio_message("assistant", msg.content))
	return gr_history, f"### 🟢 {model_obj.name}" if model_obj.name else "### 🟢 Chat mới"


def answer_question(message: str, history: list[dict[str, str]], room_id: str, request: gr.Request):
	history = list(history or [])
	question = message.strip()
	if not question:
		return history, "", room_id
	if not room_id:
		room_id = str(uuid.uuid4())

	metrics_context = begin_request_tracking()
	chat_config["configurable"]["thread_id"] = room_id # type: ignore
	route_type = None
	retrieved_docs_count = 0
	response_text = ""
	has_error = False
	try:
		rag_chain = create_agent_graph()
		result = rag_chain.invoke({
			"question": question,
			"messages": _history_to_langchain_messages(history),
			"args": None,
        },config = chat_config) # type: ignore
		route_type = result.get("type")
		args = result.get("args") or {}
		retrieved_docs = args.get("retrieved_docs") or []
		retrieved_docs_count = len(retrieved_docs)
		messages = result.get("messages") or []
		answer = "Mình chưa tạo được câu trả lời phù hợp."
		if messages:
			answer = getattr(messages[-1], "content", "")
		response_text = answer
		history.append(_gradio_message("user", question))
		history.append(_gradio_message("assistant", answer))
		save_history(room_id, history)
		return history, "", room_id
	except Exception as exc:
		error_msg = f"Đã có lỗi xảy ra: {exc}"
		has_error = True
		response_text = error_msg
		print(error_msg)
		history.append(_gradio_message("user", question))
		history.append(_gradio_message("assistant", error_msg))
		return history, "", room_id
	finally:
		finish_request_tracking(
			metrics_context,
			route_type=route_type,
			retrieved_docs=retrieved_docs_count,
			response_text=response_text,
			errored=has_error,
		)

