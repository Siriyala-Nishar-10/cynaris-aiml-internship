import os
from .agents import run_support_crew

def generate_draft(query: str, context: str) -> str:
    if os.getenv('USE_CREWAI', 'false').lower() == 'true':
        return run_support_crew(query, context)
    if context.startswith('No matching FAQ'):
        return 'Thanks for contacting support. I could not find a verified FAQ for this request. Please share relevant non-sensitive details so the team can investigate.'
    return ('Thanks for contacting support. Based on our available help information:\n\n' + context + '\n\nIf this does not resolve your issue, please reply with relevant details. Do not share passwords or full payment card details.')
