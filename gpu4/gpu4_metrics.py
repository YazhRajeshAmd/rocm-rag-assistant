import requests
import logging
import time
import re
import concurrent.futures
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
from transformers import GPT2Tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_tokenize(text):
    # Simple word-based tokenization
    return text.split()

# Initialize a general tokenizer (GPT-2 tokenizer is widely used and not gated)
try:
    general_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    logger.info("GPT-2 tokenizer initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize GPT-2 tokenizer: {str(e)}")
    logger.warning("Falling back to simple word-based tokenization")
    general_tokenizer = None

def count_tokens(text, model_name):
    if general_tokenizer:
        return len(general_tokenizer.encode(text))
    else:
        return len(simple_tokenize(text))

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

# Define four instances of different LLMs
llm1 = Ollama(
    model="mixtral:8x7b",
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    num_ctx=2048,
    repeat_penalty=1.1
)
llm2 = Ollama(
    model="llama3.1:8b",
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    num_ctx=2048,
    repeat_penalty=1.1
)
llm3 = Ollama(
    model="gemma2:27b",
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    num_ctx=2048,
    repeat_penalty=1.1
)
llm4 = Ollama(
    model="phi3:14b",
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    num_ctx=2048,
    repeat_penalty=1.1
)

# Create four QA chains
qa1 = create_qa_chain(llm1)
qa2 = create_qa_chain(llm2)
qa3 = create_qa_chain(llm3)
qa4 = create_qa_chain(llm4)

def process_model(qa, model_name, question):
    start_time = time.time()
    
    # Time RAG retrieval
    retrieval_start = time.time()
    context_docs = qa.retriever.get_relevant_documents(question)
    retrieval_time = time.time() - retrieval_start

    # Prepare input (question + context)
    context = "\n".join([doc.page_content for doc in context_docs])
    full_input = f"Question: {question}\nContext: {context}"
    
    # Count input tokens
    input_tokens = count_tokens(full_input, model_name)
    
    # Time model inference
    inference_start = time.time()
    response = qa(question)
    inference_time = time.time() - inference_start
    
    # Count output tokens
    output_tokens = count_tokens(response["result"], model_name)
    
    total_time = time.time() - start_time
    
    return {
        "response": response["result"],
        "metrics": {
            "model_name": model_name,  # Include model_name in the metrics
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "retrieval_time": retrieval_time,
            "inference_time": inference_time,
            "total_time": total_time,
            "tokens_per_second": output_tokens / inference_time if inference_time > 0 else 0
        }
    }

def respond(question, history):
    start_time = time.time()
    
    models = [
        (qa1, "mixtral:8x7b"),
        (qa2, "llama3.1:8b"),
        (qa3, "gemma2:27b"),
        (qa4, "phi3:14b")
    ]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_model = {executor.submit(process_model, qa, model_name, question): model_name for qa, model_name in models}
        
        responses = []
        individual_metrics = []
        
        for future in concurrent.futures.as_completed(future_to_model):
            result = future.result()
            responses.append(result["response"])
            individual_metrics.append(result["metrics"])
    
    end_time = time.time()
    
    # Calculate overall metrics
    total_input_tokens = sum(metric["input_tokens"] for metric in individual_metrics)
    total_output_tokens = sum(metric["output_tokens"] for metric in individual_metrics)
    total_time = end_time - start_time
    
    # Prepare performance metrics string
    performance_metrics = "Performance Metrics:\n"
    for metric in individual_metrics:
        performance_metrics += f"{metric['model_name']}:\n"
        performance_metrics += f"  Input Tokens: {metric['input_tokens']}\n"
        performance_metrics += f"  Output Tokens: {metric['output_tokens']}\n"
        performance_metrics += f"  Retrieval Time: {metric['retrieval_time']:.2f} seconds\n"
        performance_metrics += f"  Inference Time: {metric['inference_time']:.2f} seconds\n"
        performance_metrics += f"  Total Time: {metric['total_time']:.2f} seconds\n"
        performance_metrics += f"  Tokens per Second: {metric['tokens_per_second']:.2f}\n"
    
    performance_metrics += f"\nOverall:\n"
    performance_metrics += f"Total Input Tokens: {total_input_tokens}\n"
    performance_metrics += f"Total Output Tokens: {total_output_tokens}\n"
    performance_metrics += f"Total Time: {total_time:.2f} seconds\n"
    
    return responses, performance_metrics

# Create Gradio interface
def create_interface():
    custom_css = """
        .metrics-display textarea {
            font-family: monospace;
            background-color: #f0f0f0;
            border: 2px solid orange;
            border-radius: 10px;
            padding: 10px;
        }
        .chatbot1 .user { background-color: #FFA50022; }
        .chatbot1 .bot { border-left: 4px solid orange; }
        .chatbot2 .user { background-color: #0000FF22; }
        .chatbot2 .bot { border-left: 4px solid blue; }
        .chatbot3 .user { background-color: #00800022; }
        .chatbot3 .bot { border-left: 4px solid green; }
        .chatbot4 .user { background-color: #FF000022; }
        .chatbot4 .bot { border-left: 4px solid red; }
    """
    
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="purple",
    ).set(
        body_text_color="#000000",
        block_title_text_color="#000000",
        block_label_text_color="#000000",
        input_background_fill="#FFFFFF",
        checkbox_background_color="#FFFFFF",
        checkbox_border_color="#000000",
        checkbox_label_background_fill="#FFFFFF",
        checkbox_label_text_color="#000000",
        button_primary_background_fill="#FFA500",
        button_primary_background_fill_hover="#FFB52E",
        button_primary_text_color="#FFFFFF",
        button_secondary_background_fill="#FFFFFF",
        button_secondary_background_fill_hover="#F0F0F0",
        button_secondary_text_color="#000000",
    )

    with gr.Blocks(theme=theme, css=custom_css) as demo:
        gr.Markdown(
            """
            <h1 style="text-align: center; color: orange; font-size: 36px;">🚀 ROCm-bot: Multi-Model AI Showdown 🤖</h1>
            <h3 style="text-align: center; color: purple; font-size: 24px;">Witness the power of four AI titans battling to answer your questions!</h3>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot1 = gr.Chatbot(label="Mixtral 8x7B", height=300, elem_classes="chatbot1")
                chatbot2 = gr.Chatbot(label="Llama 3.1 8B", height=300, elem_classes="chatbot2")
                chatbot3 = gr.Chatbot(label="Gemma 2 27B", height=300, elem_classes="chatbot3")
                chatbot4 = gr.Chatbot(label="Phi 3 14B", height=300, elem_classes="chatbot4")
            
            with gr.Column(scale=1):
                metrics_display = gr.Textbox(
                    label="⚡ Performance Metrics ⚡", 
                    lines=20,
                    elem_classes="metrics-display"
                )
        
        msg = gr.Textbox(
            placeholder="🔥 Unleash your curiosity about ROCm and watch AI giants compete! 🔥",
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
            examples=[
                "How can I install ROCm?", 
                "What are the key features of ROCm?",
                "How does ROCm compare to CUDA?",
                "What deep learning frameworks are supported by ROCm?"
            ],
            inputs=msg,
        )

    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0")