from langchain.document_loaders import PyPDFLoader


loader = PyPDFLoader("data/deep_research_blog.pdf")
print(loader.load()[0])