import os
from openai.types import VectorStore
from openai import OpenAI
from openai.types.file_object import FileObject
from supabase import create_client

client = OpenAI()

class Storage:
    def __init__(self):
        self.__supabase = create_client(
            supabase_key=os.getenv("SUPABASE_KEY"),
            supabase_url=os.getenv("SUPABASE_URL"),
        )

    def create_bucket(self, bucket_name: str):
        return self.__supabase.storage.create_bucket(bucket_name)

    def get_bucket(self, bucket_name: str):
        return self.__supabase.storage.get_bucket(bucket_name)

    def list_buckets(self):
        return self.__supabase.storage.list_buckets()

    def list_files(self, bucket_name: str):
        return self.__supabase.storage.from_(bucket_name).list()
    
    def upload_file(self, bucket_name: str, folder_path: str):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:    
                    self.__supabase.storage.from_(bucket_name).upload(file, f)
    


def vector_store_setup() -> VectorStore:
    existed_vs = client.vector_stores.list()
    filtered_vs = [vs for vs in existed_vs if vs.name == "knowledge_base"]

    file_list = client.files.list()

    # Remove all files from the vector store
    for file in file_list:
        client.files.delete(
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


def retrieve_files(file_id: str) -> FileObject:
    file = client.files.retrieve(file_id=file_id)
    return file
