import os
from dotenv import load_dotenv
load_dotenv()

#This is to check whether API keys are properly set in environment. Even before executing tests/test_llm_client.py ot scripts/verify_llm_client.py, this should be executed.
#Before executing this, API Keys should be set in env
#touch .env
#echo 'GOOGLE_API_KEY=your_actual_key_here' >> .env
print("OPENAI_API_KEY - Available" if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY Not available")
print("GROQ_API_KEY - Available" if os.getenv("GROQ_API_KEY") else "GROQ_API_KEY Not available")
print("GEMINI_API_KEY - Available" if os.getenv("GEMINI_API_KEY") else "GEMINI_API_KEY Not available")