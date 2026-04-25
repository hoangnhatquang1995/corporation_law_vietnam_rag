from langchain.messages import SystemMessage

llm_system_prompt = SystemMessage(
    content = """
        Bạn là một trợ lý ảo. Bạn hãy luôn trả lời ngắn gọn, súc tích và chính xác nhất có thể.
    """
)
