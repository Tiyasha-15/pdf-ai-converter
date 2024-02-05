from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import chroma
import os
from constant import CHROMA_SETTINGS

persist_directory = 'db'

def main():
  for root, dirs, files in os.walk("docs"):
    for file in files:
      if file.endswith(".pdf"):
                   print(file)
                   loader = PDFMinerLoader(os.path.join(root, file))
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
  texts = text_splitter.split_documents(documents)
  
  embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
  
  db = chroma.from_documents(texts, embeddings, persist_directory=persist_directory,client_settings=CHROMA_SETTINGS)
  db.persist()
  db=None
  
if __name__ =="_main_":
  main()

