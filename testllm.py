import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)
print(env_path)
print(os.environ["LLM_BASE_URL"])
print(os.environ["LLM_MODEL"])

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
)
model = os.environ["LLM_MODEL"]



resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
)

print(resp.choices[0].message.content)
print(resp.usage)