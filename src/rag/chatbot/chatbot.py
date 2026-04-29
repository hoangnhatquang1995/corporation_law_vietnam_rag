from langchain_core.runnables import RunnableConfig
from rag.agent.graph import create_agent_graph
from langchain_core.messages import HumanMessage
from langchain_core.messages import AnyMessage
from typing import List, Any
import gradio as gr 

chat_config: RunnableConfig = {
	"configurable": {}
}


def answer_question(message: str, history : List[AnyMessage], request : gr.Request):
	question = message.strip()
	if not question:
		return "Bạn hãy nhập câu hỏi trước khi gửi."
	chat_config["configurable"]["thread_id"] = request.session_hash if request else "local"
	print(f"hisotory: {history}, question: {question}, thread_id: {chat_config['configurable']['thread_id']}")
	try:
		rag_chain = create_agent_graph()
		result = rag_chain.invoke({
			"question": question,
			"messages": history,
			"args": None,
        },config = chat_config)
	except Exception as exc:
		return f"Hệ thống chưa thể trả lời lúc này: {exc}"

	# answer = result.get("answer")
	# if isinstance(answer, str) and answer.strip():
	# 	return answer

	messages = result.get("messages") or []
	if messages:
		content = getattr(messages[-1], "content", "")
		if isinstance(content, str) and content.strip():
			return content

	return "Mình chưa tạo được câu trả lời phù hợp."
