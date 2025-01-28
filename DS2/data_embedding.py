from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from data_preprocessing import load_and_preprocess_pdfs
def store_embeddings_and_vectorstore(documents, index_path):
    """Create FAISS index and store embeddings using HuggingFaceInstructEmbeddings."""
    # Initialize HuggingFaceInstructEmbeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)
    
    # Create FAISS index
    vectorstore = FAISS.from_texts(documents, embeddings)
    vectorstore.save_local(index_path)
    print(f"Vectorstore saved at {index_path}")

def load_vectorstore(index_path):
    """Load FAISS index from the specified path."""

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
    
    vectorstore = FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)
    print(f"Vectorstore loaded from {index_path}")
    return vectorstore


if __name__ == "__main__":
    data_folder = "data"
    index_path = "faiss_index"

    documents = load_and_preprocess_pdfs(data_folder)

    store_embeddings_and_vectorstore(documents, index_path)

    vectorstore = load_vectorstore(index_path)
