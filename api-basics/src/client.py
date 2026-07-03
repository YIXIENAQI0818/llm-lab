import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client():
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
