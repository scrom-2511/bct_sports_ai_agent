import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.agents import create_agent

from prompt import SYSTEM_PROMPT
from tools import make_tools

dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

CHART_DIR = Path(__file__).parent / "charts"
CHART_DIR.mkdir(exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found in .env")


def build_agent(df):
    """
    Creates and returns an AI agent configured to analyze
    the provided dataset.

    Args:
        df (pandas.DataFrame): The dataset to be analyzed.
        api_key (str, optional): Google Gemini API Key.

    Returns:
        A LangChain agent with access to the dataset analysis tools.
    """
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY not found. Please provide an API key in the sidebar or .env file.")

    os.environ["GOOGLE_API_KEY"] = key

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    tools = make_tools(df, CHART_DIR)
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent

