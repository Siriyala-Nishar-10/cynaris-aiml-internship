from src.graph import build_graph, invoke_customer_message, resume_human_review
from src.evaluation import export_evaluation_dataset

def main():
    graph=build_graph()
    examples=[
      ('How do I reset my password?','customer-001'),
      ('Where can I find my invoice?','customer-002'),
      ('The application crashes during installation.','customer-003'),
      ('URGENT: I am locked out of my account.','customer-004'),
      ('Can you explain the support process?','customer-005')]
    print('W10D5 — Stateful Customer Support Agent')
    for i,(message,cid) in enumerate(examples,1):
        print(f'\n--- Example {i}/{len(examples)} ---\nCustomer: {message}')
        result,config=invoke_customer_message(graph,message,cid,cid)
        if '__interrupt__' in result:
            review=result['__interrupt__'][0].value
            print('\n[PAUSED FOR HUMAN REVIEW]')
            print(f"Category: {review['category']} | Priority: {review['priority']}\nDraft: {review['draft_response']}")
            decision=input("Enter 'approve' or revision feedback: ").strip()
            result=resume_human_review(graph,config,decision)
        print(f"Status: {result['status']}\nResponse: {result['final_response']}")
    print(f'\nEvaluation dataset: {export_evaluation_dataset()}')

if __name__=='__main__': main()
