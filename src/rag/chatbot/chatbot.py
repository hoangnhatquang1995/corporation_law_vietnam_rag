from langchain_core.runnables import RunnablePassthrough
from rag.agent.graph import create_agent_graph
from langchain_core.messages import HumanMessage

from rag.llm import llm

chat_config ={
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
			"messages": [HumanMessage(content=question)]
        },config = chat_config)
	except Exception as exc:
		return f"Hệ thống chưa thể trả lời lúc này: {exc}"

	return result.get("answer", "Mình chưa tạo được câu trả lời phù hợp.")
