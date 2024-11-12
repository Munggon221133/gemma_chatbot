import os
import time
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

print("Gemma Model ChatBot")

llm = ChatGroq(groq_api_key=groq_api_key, model="gemma-7b-it")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    ตอบคำถามตามบริบทที่ให้ไว้เท่านั้น
    โปรดตอบให้ถูกต้องที่สุดตามคำถาม
    <context>
    {context}
    <context>
    คำถาม: {input}
    """
)

# Embedding function for processing multiple PDFs in a directory
def vector_embedding():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Confirm this embeddings model supports Thai
    loader = PyPDFDirectoryLoader("./pdf")  # Load all PDFs in the "./pdf" directory
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

# Initialize vector embeddings if not already created
print("Initializing vector store database...")
vectors = vector_embedding()
print("Vector Store DB is Ready")

# Main chat loop
while True:
    prompt_input = input("คุณต้องการถามอะไรจากเอกสาร? (พิมพ์ 'exit' เพื่อออก): ")
    
    if prompt_input.lower() == 'exit':
        print("กำลังออกจากโปรแกรม")
        break
    
    # Create retrieval chain and get the response
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt_input})
    elapsed = time.process_time() - start
    
    # Display the answer
    print("คำตอบ:", response['answer'])
    print(f"เวลาในการตอบ: {elapsed:.2f} วินาที")
    
    # Display document similarity if desired
    show_similarity = input("คุณต้องการดูชิ้นเอกสารที่เกี่ยวข้องหรือไม่? (yes/no): ")
    if show_similarity.lower() == 'yes':
        for i, doc in enumerate(response["context"]):
            print(f"เอกสาร {i+1}:\n", doc.page_content)
            print("-" * 32)
