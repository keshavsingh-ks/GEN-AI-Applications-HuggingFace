import os
import torch
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub

# Check for GPU availability and set the appropriate device for computation./ it will take a lot of time on local cpu try running on good gpu capacity
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "removed because of privacy reason please replace with your api"


def init_llm():
    global llm_hub, embeddings

    model_id = "tiiuae/falcon-7b-instruct"  # You can change this to any other model you prefer this model is quite heavy for less complex tasks i would prefer falcon-3b
    llm_hub = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature": 0.5, "max_new_tokens": 250, "max_length": 600})

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"device": DEVICE}
    )

def process_document(document_path):
    global conversation_retrieval_chain

    loader = PyPDFLoader(document_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, embedding=embeddings)
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key="question"
    )

def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history

    output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    answer = output["result"]
    

    chat_history.append((prompt, answer))
    

    return answer

init_llm()


