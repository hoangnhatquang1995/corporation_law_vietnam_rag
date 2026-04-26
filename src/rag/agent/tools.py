from langchain.tools import tool 
from settings.types import VietnamLaw
from dataset.sql import SQLiteDatabase, VietnamLawModel
from dataset.vectorstore import VectorStoreDB, VectorStoreType
from dataset import vietnamLawSql

@tool
def lay_van_ban_luat_day_du(law_id: str) -> VietnamLaw | None:
    """
    Lấy nội dung đầy đủ của một văn bản luật dựa trên ID của nó.
    
    Args:
        law_id (str): ID của văn bản luật cần lấy nội dung.
    
    Returns:
        law (VietnamLaw): Một đối tượng chứa thông tin chi tiết về văn bản luật, bao gồm ID, tiêu đề, nội dung và ngày ban hành.
    """
    sql_law = vietnamLawSql.get(law_id)
    if sql_law:
        ret = VietnamLaw(
            **sql_law.model_dump()
        )
        print(f"[get_full_law] Retrieved law with ID {law_id}: {ret['title']}")
        return ret

    return None

@tool 
def tim_kiem_tren_mang(query : str) -> str:
    """
    Tìm kiếm thông tin trên mạng dựa trên một truy vấn cụ thể và trả về kết quả dưới dạng văn bản.
    
    Args:
        query (str): Truy vấn tìm kiếm, có thể là một câu hỏi hoặc một cụm từ liên quan đến luật doanh nghiệp Việt Nam.
    
    Returns:
        result (str): Kết quả tìm kiếm được trả về dưới dạng văn bản, có thể là một đoạn trích từ một trang web, một bài báo, hoặc một nguồn thông tin đáng tin cậy khác.
    """
    #TODO: implement web search functionality using an appropriate API (e.g., Google Custom Search API, Bing Search API, etc.)
    return "Chức năng tìm kiếm trên mạng chưa được triển khai. Vui lòng thử lại sau."