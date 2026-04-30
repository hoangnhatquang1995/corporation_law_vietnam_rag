import json
from typing import Any, List

import gradio as gr
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from rag.agent.graph import create_agent_graph
from langchain_core.messages import HumanMessage
from langchain_core.messages import AnyMessage
from typing import List, Any
import gradio as gr 
from dataset import chatroomSQL
from dataset.sql import ChatroomModel
import uuid


chat_config: RunnableConfig = {
	"configurable": {}
}

def save_history(room_id: str, history: List[List[str]]):
	flat_messages = []
	for user_msg, ai_msg in history:
		flat_messages.append(json.dumps({"type": "human", "content": user_msg}))
		flat_messages.append(json.dumps({"type": "ai", "content": ai_msg}))
    
	if chatroomSQL.have_record(room_id):
		chatroomSQL.update(room_id,{
			"messages": flat_messages
		})
	else:
		chat_name = history[0][0][:30] + "..." if history else "New Chat"
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
	temp_user = None

	model_obj = ChatroomModel.model_validate(record)
	lc_messages = model_obj.langchain_messages

	for msg in lc_messages:
		if msg.type == "human":
			temp_user = msg.content
		elif msg.type == "ai" and temp_user is not None:
			gr_history.append([temp_user, msg.content])
			temp_user = None
	return gr_history, f"### 🟢 {model_obj.name}" if model_obj.name else "### 🟢 Chat mới"

def answer_question(message: str, history : List[List[str]], room_id: str, request : gr.Request):
	question = message.strip()
	if not question:
		return "Bạn hãy nhập câu hỏi trước khi gửi."
	if not room_id:
		room_id = str(uuid.uuid4())

	chat_config["configurable"]["thread_id"] = room_id # type: ignore
	try:
		rag_chain = create_agent_graph()
		result = rag_chain.invoke({
			"question": question,
			"messages": history,
			"args": None,
        },config = chat_config) # type: ignore
		messages = result.get("messages") or []
		answer = "Mình chưa tạo được câu trả lời phù hợp."
		if messages:
			answer = getattr(messages[-1], "content", "")
		history.append([question, answer])
		save_history(room_id, history)
		return history, "", room_id
	except Exception as exc:
		error_msg = f"Đã có lỗi xảy ra: {exc}"
		print(error_msg)
		history.append([question, error_msg])
		return history, error_msg, room_id

