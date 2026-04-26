from typing import Annotated,TypedDict, List, Optional, Union, Any
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage
from torch import int64

class StateNode(TypedDict):
    messages : Annotated[List[AnyMessage], add_messages]
    context : Optional[str]
    args : Optional[dict[str,Any]]

class VietnamLaw (TypedDict):
    id: int  # Unique numeric document ID
    document_number: Optional[str]  # Official document number (e.g. 115/NQ-HĐBCQG)
    content: str  # Full text content of the legal document
    title: Optional[str]  # Full Vietnamese title
    url: Optional[str]  # Source URL on thuvienphapluat.vn
    legal_type: Optional[str]  # Document type (Quyết định, Công văn, Nghị quyết, …)
    legal_sectors: Optional[str]  # Pipe-separated sector/topic tags
    issuing_authority: Optional[str]  # Name of the issuing government body
    issuance_date: Optional[str]  # Issue date in DD/MM/YYYY format
    signers: Optional[str]  # Pipe-separated name:id pairs of signatories

