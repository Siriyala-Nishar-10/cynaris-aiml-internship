"""Exports reference examples for a separately configured Ragas evaluation run."""
import json
from pathlib import Path

def export_evaluation_dataset(path='results/evaluation_dataset.json'):
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
      {'user_input':'How do I reset my password?','retrieved_contexts':['Use the Forgot password link on the sign-in page.'],'response':'Use the Forgot password link on the sign-in page.','reference':'Use the Forgot password link on the sign-in page.'},
      {'user_input':'Where can I find my invoice?','retrieved_contexts':['Invoices are in Account > Billing > Invoices.'],'response':'Open Account > Billing > Invoices.','reference':'Open Account > Billing > Invoices.'}
    ]
    out.write_text(json.dumps(rows, indent=2), encoding='utf-8'); return out

if __name__ == '__main__': print(export_evaluation_dataset())
