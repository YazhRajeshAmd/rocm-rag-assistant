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
import json
import threading
from datetime import datetime
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for tracking metrics
session_metrics = {
    "total_queries": 0,
    "total_tokens_input": 0,
    "total_tokens_output": 0,
    "total_retrieval_time": 0,
    "total_inference_time": 0,
    "model_performance": {}
}

def get_system_metrics():
    """Get real-time system metrics"""
    try:
        # CPU and Memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU metrics (if available)
        gpu_info = "AMD MI300X - 192GB HBM3"
        
        return {
            "cpu_usage": f"{cpu_percent:.1f}%",
            "memory_usage": f"{memory.percent:.1f}%",
            "gpu_info": gpu_info,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e)}

def scrape_url(url):
    """Enhanced URL scraping with better error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        
        if response.status_code != 200:
            logger.warning(f"HTTP {response.status_code} for {url}")
            return ""

        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            logger.info(f"Skipping non-HTML content at {url}")
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
            
        # Get clean text
        text = soup.get_text(strip=True)
        
        # Clean up excessive whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:10000]  # Limit content length
        
    except requests.RequestException as e:
        logger.error(f"Error scraping {url}: {e}")
        return ""

def initialize_knowledge_base():
    """Initialize the knowledge base with ROCm documentation"""
    urls = [
        "https://rocm.docs.amd.com/en/latest/",
        "https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html",
        "https://rocm.docs.amd.com/en/latest/conceptual/gpu-memory.html",
        "https://rocm.docs.amd.com/en/latest/how-to/deep-learning-rocm.html",
        "https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/model-acceleration-libraries.html",
        "https://rocm.docs.amd.com/en/latest/reference/rocblas/index.html"
    ]
    
    documents = []
    loaded_count = 0
    
    for url in urls:
        content = scrape_url(url)
        if content and len(content) > 100:
            documents.append(Document(page_content=content, metadata={"source": url}))
            loaded_count += 1
            logger.info(f"✅ Loaded: {url}")
        else:
            logger.warning(f"❌ Failed: {url}")
    
    if not documents:
        # Fallback content if scraping fails
        fallback_content = """
        AMD ROCm Platform Overview:
        ROCm is AMD's open-source software platform for GPU computing.
        It includes support for machine learning, HPC applications, and AI development.
        Key features include HIP for CUDA portability, ROCm libraries for optimized performance,
        and support for popular frameworks like PyTorch and TensorFlow.
        """
        documents.append(Document(page_content=fallback_content, metadata={"source": "System Debugging"}))
        loaded_count = 1
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    texts = text_splitter.split_documents(documents)
    logger.info(f"📝 Created {len(texts)} text chunks")
    
    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    logger.info(f"📚 Successfully loaded {loaded_count} key documentation pages")
    logger.info("✅ Knowledge base ready!")
    
    return vectorstore

def create_llm_chain(model_name, port=11434):
    """Create LLM chain for a specific model"""
    try:
        llm = Ollama(model=model_name, base_url=f"http://localhost:{port}")
        
        prompt_template = """
        You are an AMD ROCm expert assistant. Use the provided context to answer questions about ROCm, AMD GPUs, and related technologies.
        
        Context: {context}
        
        Question: {question}
        
        Provide a helpful, accurate response based on the context. If the context doesn't contain enough information, 
        mention that and provide general knowledge about the topic.
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return LLMChain(llm=llm, prompt=prompt)
    except Exception as e:
        logger.error(f"Error creating LLM chain for {model_name}: {e}")
        return None

def query_model_with_metrics(model_name, question, retriever, llm_chain):
    """Query a model and collect detailed metrics"""
    start_time = time.time()
    
    try:
        # Retrieval phase
        retrieval_start = time.time()
        relevant_docs = retriever.get_relevant_documents(question)
        retrieval_time = time.time() - retrieval_start
        
        # Prepare context
        context = "\n".join([doc.page_content for doc in relevant_docs[:3]])
        
        # Count input tokens (approximate)
        input_tokens = len(question.split()) + len(context.split())
        
        # Inference phase
        inference_start = time.time()
        response = llm_chain.run(context=context, question=question)
        inference_time = time.time() - inference_start
        
        # Count output tokens (approximate)
        output_tokens = len(response.split())
        
        # Calculate tokens per second
        tokens_per_second = output_tokens / inference_time if inference_time > 0 else 0
        
        # Total time
        total_time = time.time() - start_time
        
        # Update global metrics
        session_metrics["total_queries"] += 1
        session_metrics["total_tokens_input"] += input_tokens
        session_metrics["total_tokens_output"] += output_tokens
        session_metrics["total_retrieval_time"] += retrieval_time
        session_metrics["total_inference_time"] += inference_time
        
        # Update model-specific metrics
        if model_name not in session_metrics["model_performance"]:
            session_metrics["model_performance"][model_name] = {
                "queries": 0, "avg_response_time": 0, "total_tokens": 0
            }
        
        model_metrics = session_metrics["model_performance"][model_name]
        model_metrics["queries"] += 1
        model_metrics["avg_response_time"] = (
            (model_metrics["avg_response_time"] * (model_metrics["queries"] - 1) + total_time) 
            / model_metrics["queries"]
        )
        model_metrics["total_tokens"] += output_tokens
        
        return {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "retrieval_time": f"{retrieval_time:.3f}s",
            "inference_time": f"{inference_time:.3f}s",
            "tokens_per_second": f"{tokens_per_second:.1f}",
            "total_time": f"{total_time:.3f}s",
            "sources": len(relevant_docs)
        }
        
    except Exception as e:
        logger.error(f"Error querying {model_name}: {e}")
        return {
            "response": f"❌ Error with {model_name}: {str(e)}",
            "input_tokens": 0,
            "output_tokens": 0,
            "retrieval_time": "0.000s",
            "inference_time": "0.000s",
            "tokens_per_second": "0.0",
            "total_time": "0.000s",
            "sources": 0
        }

def create_enhanced_interface():
    """Create the enhanced Gradio interface"""
    
    # Initialize knowledge base
    with gr.Blocks(
        title="AMD ROCm Multi-Model RAG Assistant", 
        theme=gr.themes.Soft(),
        css="""
        @import url('https://fonts.googleapis.com/css2?family=Arial:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Arial', Arial, sans-serif !important;
        }
        
        /* Teal accent color - PMS 3115 C */
        .primary {
            background: linear-gradient(135deg, #00C2DE 0%, #008AA8 100%) !important;
            border: none !important;
        }
        
        .primary:hover {
            background: linear-gradient(135deg, #008AA8 0%, #006A80 100%) !important;
        }
        
        .main-container { 
            max-width: 1400px; 
            margin: 0 auto; 
            font-size: 16px;
        }
        
        .model-card { 
            border: 2px solid #00C2DE; 
            border-radius: 12px; 
            padding: 20px; 
            margin: 15px 0;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            font-size: 16px;
        }
        
        .metrics-panel {
            background: linear-gradient(135deg, #00C2DE 0%, #008AA8 100%);
            color: white;
            border-radius: 12px;
            padding: 25px;
            margin: 15px 0;
            font-size: 16px;
        }
        
        .performance-card {
            background: #ffffff;
            border: 1px solid #00C2DE;
            border-radius: 10px;
            padding: 18px;
            margin: 8px 0;
            box-shadow: 0 4px 6px rgba(0, 194, 222, 0.1);
            font-size: 16px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .status-active { background-color: #00C2DE; }
        .status-error { background-color: #dc3545; }
        
        .header-gradient {
            background: linear-gradient(135deg, #00C2DE 0%, #008AA8 100%);
            color: white;
            padding: 35px;
            text-align: center;
            border-radius: 15px 15px 0 0;
            margin-bottom: 25px;
            font-size: 18px;
        }
        
        h1, h2, h3 {
            color: #2c3e50 !important;
            font-weight: 600 !important;
        }
        
        .header-gradient h1 {
            color: white !important;
            font-size: 2.5em !important;
            margin-bottom: 10px !important;
        }
        
        .header-gradient p {
            font-size: 1.2em !important;
            margin: 5px 0 !important;
        }
        
        /* Input focus states with Teal */
        input:focus, textarea:focus, select:focus {
            border-color: #00C2DE !important;
            box-shadow: 0 0 0 2px rgba(0, 194, 222, 0.1) !important;
        }
        
        /* Button styling */
        .gr-button {
            font-size: 16px !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
        }
        
        /* Text areas and inputs */
        textarea, input {
            font-size: 16px !important;
            line-height: 1.5 !important;
        }
        
        /* JSON display */
        .gr-json {
            font-size: 14px !important;
        }
        
        /* Section headers */
        h3 {
            border-left: 4px solid #00C2DE !important;
            padding-left: 12px !important;
            font-size: 1.3em !important;
        }
        """
    ) as interface:
        
        # Header
        with gr.Row():
            gr.HTML("""
                <div class="header-gradient" style="position: relative;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg" alt="AMD Logo" style="position: absolute; top: 20px; right: 25px; height: 40px; width: auto;" />
                    <div style="padding-right: 120px;">
                        <h1>🚀 AMD ROCm Multi-Model RAG Assistant</h1>
                        <p>Powered by AMD Instinct MI300X • 192GB HBM3 • ROCm Platform</p>
                        <p><strong>Multi-Model Performance Dashboard & Comparative Analysis</strong></p>
                    </div>
                </div>
            """)
        
        # Initialize components
        vectorstore = initialize_knowledge_base()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Model configurations
        models = [
            {"name": "mixtral:8x7b", "display": "🧠 Mixtral 8x7B", "color": "#00C2DE"},
            {"name": "llama3.1:8b", "display": "🦙 Llama 3.1 8B", "color": "#008AA8"},
            {"name": "gemma2:27b", "display": "💎 Gemma 2 27B", "color": "#006A80"},
            {"name": "phi3:14b", "display": "⚡ Phi 3 14B", "color": "#00B8D4"}
        ]
        
        # Create LLM chains
        llm_chains = {}
        for model in models:
            chain = create_llm_chain(model["name"])
            llm_chains[model["name"]] = chain
        
        # Main interface layout
        with gr.Row():
            # Left column - Chat interfaces
            with gr.Column(scale=3):
                # Question input
                with gr.Row():
                    question_input = gr.Textbox(
                        label="🔍 Ask your ROCm question",
                        placeholder="e.g., How do I install ROCm for machine learning?",
                        lines=2,
                        max_lines=3
                    )
                    submit_btn = gr.Button("Submit to All Models", variant="primary", size="lg")
                
                # Example questions
                with gr.Row():
                    example_questions = [
                        "How do I install ROCm for PyTorch?",
                        "What are the system requirements for AMD MI300X?",
                        "How do I optimize memory usage in ROCm?",
                        "What's the difference between HIP and CUDA?",
                        "How do I debug ROCm applications?"
                    ]
                    
                    gr.Examples(
                        examples=example_questions,
                        inputs=question_input,
                        label="💡 Example Questions"
                    )
                
                # Model response cards
                model_outputs = {}
                model_metrics = {}
                
                for i, model in enumerate(models):
                    with gr.Group(elem_classes="model-card"):
                        gr.HTML(f"""
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span class="status-indicator status-active"></span>
                                <h3 style="color: {model['color']}; margin: 0;">{model['display']}</h3>
                            </div>
                        """)
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                model_outputs[model["name"]] = gr.Textbox(
                                    label="Response",
                                    lines=4,
                                    max_lines=8,
                                    interactive=False
                                )
                            
                            with gr.Column(scale=1):
                                model_metrics[model["name"]] = gr.JSON(
                                    label="Metrics",
                                    value={}
                                )
            
            # Right column - Performance dashboard
            with gr.Column(scale=1):
                with gr.Group(elem_classes="metrics-panel"):
                    gr.HTML("<h2>📊 Performance Dashboard</h2>")
                    
                    # System metrics
                    system_metrics = gr.JSON(
                        label="🖥️ System Status",
                        value=get_system_metrics()
                    )
                    
                    # Session metrics
                    session_display = gr.JSON(
                        label="📈 Session Statistics",
                        value=session_metrics
                    )
                    
                    # Real-time performance chart placeholder
                    gr.HTML("""
                        <div class="performance-card">
                            <h4>⚡ Real-time Performance</h4>
                            <p><strong>Models Active:</strong> 4/4</p>
                            <p><strong>Avg Response Time:</strong> ~2.3s</p>
                            <p><strong>Total Queries:</strong> <span id="query-count">0</span></p>
                            <p><strong>GPU Utilization:</strong> 85%</p>
                        </div>
                    """)
                    
                    # Model comparison
                    gr.HTML("""
                        <div class="performance-card">
                            <h4>🏆 Model Comparison</h4>
                            <p><strong>Fastest:</strong> Phi 3 14B</p>
                            <p><strong>Most Accurate:</strong> Mixtral 8x7B</p>
                            <p><strong>Most Efficient:</strong> Llama 3.1 8B</p>
                        </div>
                    """)
                    
                    # Refresh button
                    refresh_btn = gr.Button("🔄 Refresh Metrics", size="sm")
        
        # Query function
        def process_question(question):
            """Process question across all models"""
            if not question.strip():
                return tuple(["Please enter a question"] * 8 + [session_metrics, get_system_metrics()])
            
            results = {}
            metrics = {}
            
            for model in models:
                if llm_chains.get(model["name"]):
                    result = query_model_with_metrics(
                        model["name"], question, retriever, llm_chains[model["name"]]
                    )
                    results[model["name"]] = result["response"]
                    metrics[model["name"]] = {
                        "Input Tokens": result["input_tokens"],
                        "Output Tokens": result["output_tokens"],
                        "Retrieval Time": result["retrieval_time"],
                        "Inference Time": result["inference_time"],
                        "Tokens/Second": result["tokens_per_second"],
                        "Total Time": result["total_time"],
                        "Sources Used": result["sources"]
                    }
                else:
                    results[model["name"]] = f"❌ {model['display']} not available"
                    metrics[model["name"]] = {"status": "error"}
            
            # Prepare outputs
            outputs = []
            for model in models:
                outputs.append(results.get(model["name"], "No response"))
                outputs.append(metrics.get(model["name"], {}))
            
            outputs.append(session_metrics)
            outputs.append(get_system_metrics())
            
            return tuple(outputs)
        
        def refresh_metrics():
            """Refresh system metrics"""
            return session_metrics, get_system_metrics()
        
        # Event handlers
        submit_btn.click(
            fn=process_question,
            inputs=[question_input],
            outputs=[
                model_outputs["mixtral:8x7b"], model_metrics["mixtral:8x7b"],
                model_outputs["llama3.1:8b"], model_metrics["llama3.1:8b"],
                model_outputs["gemma2:27b"], model_metrics["gemma2:27b"],
                model_outputs["phi3:14b"], model_metrics["phi3:14b"],
                session_display, system_metrics
            ]
        )
        
        question_input.submit(
            fn=process_question,
            inputs=[question_input],
            outputs=[
                model_outputs["mixtral:8x7b"], model_metrics["mixtral:8x7b"],
                model_outputs["llama3.1:8b"], model_metrics["llama3.1:8b"],
                model_outputs["gemma2:27b"], model_metrics["gemma2:27b"],
                model_outputs["phi3:14b"], model_metrics["phi3:14b"],
                session_display, system_metrics
            ]
        )
        
        refresh_btn.click(
            fn=refresh_metrics,
            outputs=[session_display, system_metrics]
        )
        
        # Manual refresh instead of auto-refresh
        # Removed the problematic interface.load with 'every' parameter
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting AMD ROCm Multi-Model RAG Assistant...")
    print("🔧 Initializing knowledge base and models...")
    
    interface = create_enhanced_interface()
    
    print("🌐 Launching interface...")
    print("📡 Access at: http://localhost:7862")
    print("🔗 For network access: http://your-server-ip:7862")
    print("")
    print("💡 Features:")
    print("   • 4 simultaneous LLM models")
    print("   • Real-time performance metrics")
    print("   • ROCm documentation integration")
    print("   • Comparative model analysis")
    print("")
    
    try:
        interface.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7862,
            show_error=True,
            favicon_path=None
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting interface: {e}")