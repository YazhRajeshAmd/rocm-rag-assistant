import requests
import logging
import time
import subprocess
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from bs4 import BeautifulSoup
import gradio as gr
from urllib.parse import urljoin

# Function to scrape content from a URL
def scrape_url(url):
    try:
        response = requests.get(url, timeout=10)
        
        # Check if the content is HTML
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            print(f"Skipping non-HTML content at {url}")
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(strip=True)
    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Main website URL
main_url = "https://rocm.docs.amd.com/en/latest/"

# Scrape the main page
try:
    main_page = requests.get(main_url)
    main_soup = BeautifulSoup(main_page.text, 'html.parser')
except requests.RequestException as e:
    print(f"Error accessing main page: {e}")
    exit(1)

# Find all links on the main page
links = main_soup.find_all('a', href=True)

# Scrape content from the main page and linked pages
all_content = []
all_content.append(Document(page_content=scrape_url(main_url), metadata={"source": main_url}))
for link in links:
    full_url = urljoin(main_url, link['href'])
    if (full_url == "https://www.amd.com/"):
        print("detected main amd site-ignoring")
    elif (full_url == "https://www.amd.com/en/developer/resources/infinity-hub.html"):
        print("detected Infinity Hub-ignoring")
    elif full_url.startswith('http') and not full_url.lower().endswith('.pdf'):
        content = scrape_url(full_url)
        print(f"Scraping site: ", full_url)
        if content:
            all_content.append(Document(page_content=content, metadata={"source": full_url}))

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(all_content)

# Create embeddings
embeddings = HuggingFaceEmbeddings()

# Create a vector store
vectorstore = Chroma.from_documents(texts, embeddings)

# Create a retriever
retriever = vectorstore.as_retriever()

# Function to start Ollama instances
def start_ollama_instances(num_instances=4, base_port=11434):
    processes = []
    for i in range(num_instances):
        port = base_port + i
        cmd = f"ollama serve -p {port}"
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)
    return processes

# Start Ollama instances
ollama_processes = start_ollama_instances()

# Define four instances of llama3.1:8b, each on a different port
llm1 = Ollama(model="llama3.1:8b", base_url="http://localhost:11434")
llm2 = Ollama(model="llama3.1:8b", base_url="http://localhost:11435")
llm3 = Ollama(model="llama3.1:8b", base_url="http://localhost:11436")
llm4 = Ollama(model="llama3.1:8b", base_url="http://localhost:11437")

# Define the prompt
prompt = """
1. Use the following pieces of context to answer the question related to AMD's ROCm product.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences, and try to only use the context provided when answering.

Context: {context}

Question: {question}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) 

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

# Function to create QA chain
def create_qa_chain(llm):
    llm_chain = LLMChain(
        llm=llm, 
        prompt=QA_CHAIN_PROMPT, 
        callbacks=None, 
        verbose=True
    )
    
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None
    )
    
    return RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        verbose=True,
        retriever=retriever,
        return_source_documents=True
    )

# Create four QA chains
qa1 = create_qa_chain(llm1)
qa2 = create_qa_chain(llm2)
qa3 = create_qa_chain(llm3)
qa4 = create_qa_chain(llm4)

# Function to get responses from all four LLMs and calculate performance metrics
def respond(question, history):
    start_time = time.time()
    
    responses = []
    individual_metrics = []
    
    for qa in [qa1, qa2, qa3, qa4]:
        instance_start_time = time.time()
        response = qa(question)
        instance_end_time = time.time()
        
        instance_time = instance_end_time - instance_start_time
        instance_tokens = len(response["result"].split())  # Approximation
        instance_tokens_per_second = instance_tokens / instance_time
        
        responses.append(response["result"])
        individual_metrics.append({
            "tokens": instance_tokens,
            "time": instance_time,
            "tokens_per_second": instance_tokens_per_second
        })
    
    end_time = time.time()
    
    # Calculate overall metrics
    total_tokens = sum(metric["tokens"] for metric in individual_metrics)
    total_time = end_time - start_time
    average_tokens_per_second = total_tokens / total_time
    
    # Prepare performance metrics string
    performance_metrics = "Performance Metrics:\n"
    for i, metric in enumerate(individual_metrics, 1):
        performance_metrics += f"Instance {i}:\n"
        performance_metrics += f"  Tokens: {metric['tokens']}\n"
        performance_metrics += f"  Time: {metric['time']:.2f} seconds\n"
        performance_metrics += f"  Tokens per second: {metric['tokens_per_second']:.2f}\n"
    
    performance_metrics += f"\nOverall:\n"
    performance_metrics += f"Total tokens: {total_tokens}\n"
    performance_metrics += f"Total time: {total_time:.2f} seconds\n"
    performance_metrics += f"Average tokens per second: {average_tokens_per_second:.2f}"
    
    return responses, performance_metrics

# Create Gradio interface
def create_interface():
    with gr.Blocks(theme="glass") as demo:
        gr.Markdown("# Rocm-bot (Quad Llama 3.1 8B)")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot1 = gr.Chatbot(label="Llama 3.1 8B Instance 1", height=300)
                chatbot2 = gr.Chatbot(label="Llama 3.1 8B Instance 2", height=300)
                chatbot3 = gr.Chatbot(label="Llama 3.1 8B Instance 3", height=300)
                chatbot4 = gr.Chatbot(label="Llama 3.1 8B Instance 4", height=300)
            
            with gr.Column(scale=1):
                metrics_display = gr.Textbox(label="Performance Metrics", lines=20)
        
        msg = gr.Textbox(
            placeholder="Ask me questions related to the awesomeness of ROCm and how it can revolutionize your AI workflows",
            container=False,
            scale=7
        )

        def user(user_message, history1, history2, history3, history4):
            responses, metrics = respond(user_message, None)
            history1.append((user_message, responses[0]))
            history2.append((user_message, responses[1]))
            history3.append((user_message, responses[2]))
            history4.append((user_message, responses[3]))
            return "", history1, history2, history3, history4, metrics

        msg.submit(user, [msg, chatbot1, chatbot2, chatbot3, chatbot4], [msg, chatbot1, chatbot2, chatbot3, chatbot4, metrics_display])

        gr.Examples(
            examples=["How can I install ROCm", "What installation methods exist for ROCm"],
            inputs=msg,
        )

    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    try:
        demo.launch(share=True, server_name="0.0.0.0", server_port=7862)
    finally:
        # Clean up Ollama processes
        for process in ollama_processes:
            process.terminate()
        for process in ollama_processes:
            process.wait()