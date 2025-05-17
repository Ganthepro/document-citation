

from openai import OpenAI
client = OpenAI()


import asyncio

async def create_file() -> str:
    def read_file():
        with open("data/deep_research_blog.pdf", "rb") as file_content:
            return client.files.create(
                file=file_content,
                purpose="assistants"
            )
    result = await asyncio.to_thread(read_file)
    print("id", result.id)
    print("filename", result.filename)
    return result.id
