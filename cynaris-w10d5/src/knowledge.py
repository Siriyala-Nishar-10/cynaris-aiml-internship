import json
from pathlib import Path
FAQ_PATH = Path(__file__).resolve().parents[1] / 'data' / 'support_faq.json'

def retrieve_faq(query: str) -> str:
    words = {w.lower().strip('.,?!') for w in query.split() if len(w) > 2}
    items = json.loads(FAQ_PATH.read_text(encoding='utf-8'))
    ranked = []
    for item in items:
        haystack = (item['question'] + ' ' + item['answer']).lower()
        score = sum(word in haystack for word in words)
        if score:
            ranked.append((score, item))
    if not ranked:
        return 'No matching FAQ found. Ask the support team to verify the applicable policy.'
    ranked.sort(key=lambda x: x[0], reverse=True)
    return '\n'.join(f"Q: {i['question']}\nA: {i['answer']}" for _, i in ranked[:2])
