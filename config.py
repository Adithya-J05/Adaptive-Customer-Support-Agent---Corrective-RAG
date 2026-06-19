import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def load_env() -> None:
    load_dotenv()
    if "GOOGLE_API_KEY" not in os.environ:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Either:\n"
            "  - Create a .env file with GOOGLE_API_KEY=your_key\n"
            "  - export GOOGLE_API_KEY=your_key"
        )


def get_llm():
    primary = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        max_retries=1,
    )
    fallback = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0,
        max_retries=2,
    )
    return primary.with_fallbacks([fallback])


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
