import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")
print(f"Loaded API Key: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")

try:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=api_key
    )
    print("Attempting to invoke Groq API...")
    response = llm.invoke("Hello, are you working?")
    print(f"✅ Success! Response: {response.content}")
except Exception as e:
    print(f"❌ Error: {e}")
