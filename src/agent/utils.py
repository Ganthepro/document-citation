import os
from openai.types import VectorStore
from openai import OpenAI

client = OpenAI()

def vector_store_setup() -> VectorStore:
    vector_store = client.vector_stores.create(name="knowledge_base")


    # Clear the vector store
    file_list = client.files.list()
    for file in file_list:
        client.files.delete(
            file_id=file.id,
        )

    # Define the source folder path
    folder_path = "data"  

    # Upload files to the vector store
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as file_content:
                    result = client.files.create(file=file_content, purpose="assistants")
                    client.vector_stores.files.create(
                        vector_store_id=vector_store.id,
                        file_id=result.id,
                    )
    except Exception as e:
        raise FileNotFoundError(
            f"File not found: {e}"
        ) from e
    return vector_store