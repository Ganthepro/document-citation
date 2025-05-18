import os
from openai.types import VectorStore
from openai import OpenAI

client = OpenAI()


def vector_store_setup() -> VectorStore:
    existed_vs = client.vector_stores.list()
    filtered_vs = [
        vs for vs in existed_vs if vs.name == "knowledge_base"
    ]

    # Remove all files from the vector store
    for vs in filtered_vs:
        files = client.vector_stores.files.list(vector_store_id=vs.id)
        for file in files:
            client.vector_stores.files.delete(
                vector_store_id=vs.id,
                file_id=file.id,
            )
    
    # Remove all vector stores with the name "knowledge_base"
    for vs in filtered_vs:
        client.vector_stores.delete(
            vector_store_id=vs.id,    
        )

    # Create a new vector store
    vector_store = client.vector_stores.create(name="knowledge_base")

    # Define the source folder path
    folder_path = "data"

    # Upload files to the vector store
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as file_content:
                    result = client.files.create(
                        file=file_content, purpose="assistants"
                    )
                    client.vector_stores.files.create(
                        vector_store_id=vector_store.id,
                        file_id=result.id,
                    )
    except Exception as e:
        raise FileNotFoundError(f"File not found: {e}") from e
    return vector_store
