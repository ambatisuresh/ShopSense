# scripts/verify_llm_client.py
import os
from dotenv import load_dotenv
load_dotenv()

from core.llm_client import LLMClient

#This is a basic verification that LLMs are working fine and have valid api key
def test_basic_completion():
    client = LLMClient()
    response = client.complete([
        {"role": "user", "content": "Reply with exactly one word: 'Jai Balayya'"}
    ])
    print(f"Model used: {client.model}")
    print(f"Response: {response!r}")
    assert isinstance(response, str), "complete() should return a str"
    assert len(response) > 0, "response should not be empty"
    print("✅ basic completion works")

def test_system_message_respected():
    client = LLMClient()
    response = client.complete([
        {"role": "system", "content": "You always answer in French, no matter what."},
        {"role": "user", "content": "Say hello"},
    ])
    print(f"Response: {response!r}")
    print("✅ Manually verify that hello is said in French. If yes, it is respecting system messages.")

def test_bad_api_key_fails_loudly():
    client = LLMClient(api_key="invalid-key-123")
    try:
        client.complete([{"role": "user", "content": "hi"}])
        print("❌ expected an exception with a bad API key, got none")
    except Exception as e:
        print(f"✅ correctly raised: {type(e).__name__}: {e}")

if __name__ == "__main__":
    print(f"\n{'─' * 50}")
    print("Checking Basic completion")
    print(f"{'─' * 50}")
    test_basic_completion()
    print(f"\n{'─' * 50}")
    print("Checking if system messages are respected by LLM or not")
    print(f"{'─' * 50}")
    test_system_message_respected()
    print(f"\n{'─' * 50}")
    print("Checking Invalid Key Authentication: Should throw error")
    print(f"{'─' * 50}")
    test_bad_api_key_fails_loudly()