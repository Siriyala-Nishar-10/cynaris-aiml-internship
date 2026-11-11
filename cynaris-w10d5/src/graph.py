from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from .state import SupportState
from .knowledge import retrieve_faq
from .crew import generate_draft
from .tracking import log_run

def classify(state: SupportState) -> dict:
    query = state['messages'][-1].content.lower()
    if any(w in query for w in ('refund','payment','billing','charge','invoice','account','login','password')):
        category = 'account_billing'
    elif any(w in query for w in ('error','crash','install','bug','technical','code','app')):
        category = 'technical'
    else:
        category = 'general'
    priority = 'high' if any(w in query for w in ('urgent','immediately','locked out','fraud','unauthorized')) else 'normal'
    return {'category': category, 'priority': priority, 'status': 'classified'}

def retrieve_context(state: SupportState) -> dict:
    return {'faq_context': retrieve_faq(state['messages'][-1].content)}

def draft_response(state: SupportState) -> dict:
    query = state['messages'][-1].content
    return {'draft_response': generate_draft(query, state.get('faq_context','')), 'needs_human': state.get('priority') == 'high' or state.get('category') == 'account_billing'}

def review_response(state: SupportState) -> dict:
    decision = interrupt({'message':'Human review required before sending.', 'category':state.get('category'), 'priority':state.get('priority'), 'draft_response':state.get('draft_response'), 'instructions':"Reply 'approve' or provide revision feedback."})
    if isinstance(decision, str) and decision.strip().lower() in {'approve','approved','yes'}:
        final, status, note = state['draft_response'], 'approved', 'Approved by human reviewer.'
    else:
        final, status, note = 'Your request has been sent to a support specialist for review. The team will follow up after checking the details.', 'escalated', str(decision)
    log_run(state.get('category','unknown'), state.get('priority','normal'), status)
    return {'final_response':final, 'review_note':note, 'status':status}

def finish_without_review(state: SupportState) -> dict:
    log_run(state.get('category','unknown'), state.get('priority','normal'), 'completed')
    return {'final_response':state['draft_response'], 'status':'completed'}

def build_graph():
    b = StateGraph(SupportState)
    b.add_node('classify', classify); b.add_node('retrieve_context', retrieve_context)
    b.add_node('draft_response', draft_response); b.add_node('review_response', review_response)
    b.add_node('finish_without_review', finish_without_review)
    b.add_edge(START,'classify'); b.add_edge('classify','retrieve_context'); b.add_edge('retrieve_context','draft_response')
    b.add_conditional_edges('draft_response', lambda s: 'review' if s.get('needs_human') else 'finish', {'review':'review_response','finish':'finish_without_review'})
    b.add_edge('review_response',END); b.add_edge('finish_without_review',END)
    return b.compile(checkpointer=MemorySaver())

def invoke_customer_message(graph, message: str, thread_id: str, customer_id: str='demo-customer'):
    config = {'configurable': {'thread_id':thread_id}}
    result = graph.invoke({'messages':[{'role':'user','content':message}], 'customer_id':customer_id}, config=config)
    return result, config

def resume_human_review(graph, config: dict, decision: str):
    return graph.invoke(Command(resume=decision), config=config)
