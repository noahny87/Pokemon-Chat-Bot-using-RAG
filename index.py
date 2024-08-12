from langchain_community.document_loaders.csv_loader import CSVLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_community.llms import ctransformers
from langchain.memory import ConversationBufferMemory 
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from huggingface_hub import login
import pandas as pd
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
#set 4bit quantization bit 
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
#load csv file
loader = CSVLoader(file_path = "pokemon.csv", encoding = "utf-8", csv_args = {'delimiter':","})
data = loader.load()
#split data into chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)
text_chunks = text_splitter.split_documents(data)

#load embeddings from hugging face using llama 2 
embeddings = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
#store embeddings in vector store 
docsearch = FAISS.from_documents(text_chunks, embeddings)
#save vector store 
docsearch.save_local("vectorstore")
#create sample query 

#tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
#model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B",device_map = "auto", torch_dtype = torch.bfloat16 )

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map = "auto", torch_dtype = torch.bfloat16, quantization_config = config, low_cpu_mem_usage = True)

pipe = pipeline("text-generation",model=model, tokenizer=tokenizer)
# Create a conversation buffer memory
memory = ConversationBufferMemory()

# Function to chat with the model
def chat_with_model(Context1, query):
    # Retrieve similar documents using docsearch (you can add this part)
    docs = docsearch.similarity_search(Context1, k=20)
    
    # Concatenate the retrieved documents
    context = " ".join([doc.page_content for doc in docs])
    
    # Prepare the input for the language model
    input_text = f"Here is the context to the question the user asked: {context}\n Here is the question the user asked: {query}" 
    # Note: You can adjust padding and truncation options as needed
    
    # Generate response
    outputs = pipe(input_text, max_new_tokens=2000)[0]["generated_text"][len(input_text):]
    
    return outputs

# Test the chat function
Context = "fire type and Legendary"
query = "What is the type of the first pokemon"

output = chat_with_model(Context, query)
print(output)




