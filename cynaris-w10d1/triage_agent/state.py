import operator
from typing import Annotated, Optional, TypedDict


class AgentState(TypedDict, total=False):
    message: str
    label: str                        # billing | technical | refund | general | unknown
    confidence: Optional[float]       # None = the classifier gives no confidence signal
    method: str                       # which classifier produced the label
    route: str                        # "auto" or "human"
    route_reason: str                 # why: shown to the human reviewer
    human_decision: Optional[dict]    # {"approved": bool, "note": str}
    response: str
    # Reducer: each node APPENDS its name, so the final state records the exact
    # path taken. Tests assert on this instead of guessing from the output text.
    trace: Annotated[list, operator.add]
