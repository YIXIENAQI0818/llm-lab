import os
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client():
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
