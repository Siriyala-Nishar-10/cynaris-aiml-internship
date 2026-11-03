"""A support-ticket triage agent built as a LangGraph state machine.

    classify -> route -+-> respond                       (confident, low-risk)
                       +-> human_review -> respond       (refunds, unclear, mixed)
"""
