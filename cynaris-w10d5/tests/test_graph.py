from src.graph import build_graph, invoke_customer_message, resume_human_review

def test_general_request_completes_without_review():
    g=build_graph(); r,_=invoke_customer_message(g,'Can you explain your help center?','general-1')
    assert r['category']=='general' and r['status']=='completed' and r['final_response']

def test_billing_request_pauses_and_resumes_after_approval():
    g=build_graph(); r,c=invoke_customer_message(g,'I need help with my billing details','billing-1')
    assert '__interrupt__' in r
    done=resume_human_review(g,c,'approve')
    assert done['status']=='approved' and done['final_response']

def test_urgent_request_escalates_on_revision_feedback():
    g=build_graph(); r,c=invoke_customer_message(g,'URGENT: I am locked out of my account','urgent-1')
    assert '__interrupt__' in r
    done=resume_human_review(g,c,'Please escalate to a specialist')
    assert done['status']=='escalated'
