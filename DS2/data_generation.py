import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from data_embedding import load_vectorstore

# Function to generate responses using LLaMA 2 with CTransformers and RetrievalQA
def generate_response(user_query, vectorstore_path):
    # Step 1: Load the FAISS vectorstore
    print("Loading vector store...")
    vectorstore = load_vectorstore(vectorstore_path)

  
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    
    print("Loading LLaMA 2 model via CTransformers...")
    model_path = "llama-2-7b-chat.ggmlv3.q8_0.bin"  # Update this path
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        max_new_tokens=1040,
        temperature=0.7,
        top_p=0.95,
        device="cpu",  # Set to 'cuda' if GPU is available and supported
    )

   
    prompt_template = """
    Use the following context to answer the question:
    {context}
    
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

  
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # You can change this to other types like "map_reduce", "refine"
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={'prompt':prompt}
    )

   
    print("Generating response...")
    response = qa_chain.run(user_query)
    return response


if __name__ == "__main__":

    VECTORSTORE_PATH = "faiss_index"
    USER_QUERY = "Can you summarize the contents of the documents?"

    # Generate response
    result = generate_response(USER_QUERY, VECTORSTORE_PATH)
    print("\nGenerated Response:")
    print(result)
