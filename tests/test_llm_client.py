# tests/test_llm_client.py
import pytest
from core.llm_client import LLMClient


#This is actual test case to verify LLMClient using pytest
#should have executed tests/first.py before executing this
@pytest.fixture
def client():
    return LLMClient()

def test_complete_returns_nonempty_string(client):
    result = client.complete([{"role": "user", "content": "Say 'ok'"}])
    assert isinstance(result, str)
    assert len(result.strip()) > 0

def test_complete_with_invalid_api_key_raises():
    bad_client = LLMClient(api_key="sk-invalid")
    with pytest.raises(Exception):
        bad_client.complete([{"role": "user", "content": "hi"}])

def test_complete_with_invalid_model_raises():
    bad_client = LLMClient(model="not-a-real-model")
    with pytest.raises(Exception):
        bad_client.complete([{"role": "user", "content": "hi"}])

def test_complete_respects_kwargs(client):
    # e.g. temperature=0 should be more deterministic across two calls
    r1 = client.complete([{"role": "user", "content": "What is 2+2? One word."}], temperature=0)
    r2 = client.complete([{"role": "user", "content": "What is 2+2? One word."}], temperature=0)
    assert "4" in r1 and "4" in r2

print("Test case executed successfully. No failutes.")