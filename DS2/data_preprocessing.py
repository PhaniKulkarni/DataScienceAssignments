import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


def load_and_preprocess_pdfs(data_folder):
    split_documents = []
    
    # Loop through all files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            
            loader = PyPDFLoader(file_path=pdf_path)
            documents = loader.load()
            
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
            split_docs = text_splitter.split_documents(documents)

            cleaned_chunks = [clean_text(doc.page_content) for doc in split_docs]

            split_documents.extend(cleaned_chunks)
    
    return split_documents


def clean_text(text):
    cleaned_text = ' '.join(text.split())
    return cleaned_text

# Folder containing PDF files
data_folder = "data"

split_documents = load_and_preprocess_pdfs(data_folder)


print(split_documents[:5])  
