from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class SupportState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    customer_id: str
    category: str
    priority: str
    faq_context: str
    draft_response: str
    final_response: str
    needs_human: bool
    review_note: str
    status: str
