from src.knowledge import retrieve_faq

def test_password_faq_is_retrieved():
    assert 'Forgot password' in retrieve_faq('How do I reset my password?')

def test_unknown_query_has_safe_fallback():
    assert 'No matching FAQ' in retrieve_faq('Explain an unrelated astronomy topic')
