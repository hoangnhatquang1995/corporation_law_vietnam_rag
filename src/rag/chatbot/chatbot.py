from langchain_core.runnables import RunnableConfig
from rag.agent.graph import create_agent_graph
from langchain_core.messages import HumanMessage

chat_config: RunnableConfig = {
	"configurable": {
		"thread_id": "phong_chat_001"}
}

def answer_question(message: str, history):
	question = message.strip()
	if not question:
		return "Bạn hãy nhập câu hỏi trước khi gửi."

	try:
		rag_chain = create_agent_graph()
		result = rag_chain.invoke({
			"messages": [HumanMessage(content=question)],
			"args": None,
			"context": None
        },config = chat_config)
	except Exception as exc:
		return f"Hệ thống chưa thể trả lời lúc này: {exc}"

	answer = result.get("answer")
	if isinstance(answer, str) and answer.strip():
		return answer

	messages = result.get("messages") or []
	if messages:
		content = getattr(messages[-1], "content", "")
		if isinstance(content, str) and content.strip():
			return content

	return "Mình chưa tạo được câu trả lời phù hợp."
