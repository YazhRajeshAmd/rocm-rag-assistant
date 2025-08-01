#!/usr/bin/env python3
"""
ROCm Installation & Troubleshooting Assistant
Enhanced version with modern UI and focused use case
"""

import requests
import time
import subprocess
import os
from urllib.parse import urljoin
import gradio as gr
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

class ROCmAssistant:
    def __init__(self):
        self.qa_chains = []
        self.setup_complete = False
        
    def scrape_rocm_docs(self):
        """Scrape ROCm documentation for installation and troubleshooting"""
        print("🔍 Gathering ROCm documentation...")
        
        # Focus on key installation and troubleshooting pages
        key_urls = [
            "https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html",
            "https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html", 
            "https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html",
            "https://rocm.docs.amd.com/en/latest/how-to/system-debugging.html",
            "https://rocm.docs.amd.com/en/latest/deploy/linux/prerequisites.html",
            "https://rocm.docs.amd.com/en/latest/reference/gpu-arch.html"
        ]
        
        documents = []
        for url in key_urls:
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    content = soup.get_text(strip=True)
                    if content and len(content) > 500:
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": url,
                                "title": self._extract_title(url),
                                "type": "installation" if "install" in url else "troubleshooting"
                            }
                        )
                        documents.append(doc)
                        print(f"✅ Loaded: {self._extract_title(url)}")
                    else:
                        print(f"⚠️ Skipped: {url} (insufficient content)")
                else:
                    print(f"❌ Failed: {url} (HTTP {response.status_code})")
            except Exception as e:
                print(f"❌ Error loading {url}: {str(e)[:100]}")
        
        print(f"📚 Successfully loaded {len(documents)} key documentation pages")
        return documents
    
    def _extract_title(self, url):
        """Extract readable title from URL"""
        parts = url.split('/')
        if len(parts) > 0:
            title = parts[-1].replace('.html', '').replace('_', ' ').replace('-', ' ')
            return title.title()
        return "ROCm Documentation"
    
    def setup_knowledge_base(self):
        """Setup the vector database and retrieval system"""
        print("🧠 Setting up knowledge base...")
        
        # Get documents
        documents = self.scrape_rocm_docs()
        if not documents:
            raise Exception("Failed to load any documentation")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📝 Created {len(chunks)} text chunks")
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory="./rocm_kb",
            collection_name="rocm_install_troubleshoot"
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7}
        )
        
        print("✅ Knowledge base ready!")
        return retriever
    
    def setup_llm_chain(self, retriever):
        """Setup multiple LLM chains for better performance"""
        print("🤖 Connecting to multiple Ollama instances...")
        
        # Check available Ollama instances (using different ports to avoid conflicts)
        ports = [11438, 11439, 11440, 11441]
        available_llms = []
        
        for port in ports:
            try:
                response = requests.get(f"http://localhost:{port}/api/tags", timeout=3)
                if response.status_code == 200:
                    llm = Ollama(model="llama3.1:8b", base_url=f"http://localhost:{port}")
                    available_llms.append(llm)
                    print(f"✅ Connected to Ollama on port {port}")
                else:
                    print(f"⚠️  Ollama on port {port} not responding properly")
            except Exception as e:
                print(f"❌ Ollama on port {port} not accessible")
        
        if not available_llms:
            raise Exception("No Ollama instances found. Please start Ollama servers.")
        
        print(f"🎉 Successfully connected to {len(available_llms)} Ollama instances")
        
        # Enhanced prompt for installation and troubleshooting
        prompt_template = """You are a ROCm Installation and Troubleshooting Expert. Your specialty is helping users successfully install, configure, and troubleshoot AMD ROCm on Linux systems.

CONTEXT: {context}

QUESTION: {question}

Please provide a detailed, step-by-step response following these guidelines:

1. **Be Specific**: Give exact commands, file paths, and configuration details
2. **Include Prerequisites**: Mention system requirements and dependencies
3. **Provide Alternatives**: Offer multiple solutions when applicable
4. **Add Warnings**: Highlight potential issues or important notes
5. **Verify Steps**: Include verification commands to check if steps worked
6. **Troubleshoot**: If this is an error, provide diagnostic steps

Structure your response with:
- **Overview** (brief summary)
- **Step-by-step Instructions** (numbered steps)
- **Verification** (how to confirm it worked)
- **Common Issues** (potential problems and solutions)

EXPERT RESPONSE:"""

        qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Create multiple QA chains
        qa_chains = []
        for i, llm in enumerate(available_llms, 1):
            try:
                llm_chain = LLMChain(llm=llm, prompt=qa_prompt, verbose=False)
                
                doc_prompt = PromptTemplate(
                    input_variables=["page_content", "source"],
                    template="Source: {source}\nContent: {page_content}"
                )
                
                combine_chain = StuffDocumentsChain(
                    llm_chain=llm_chain,
                    document_variable_name="context",
                    document_prompt=doc_prompt
                )
                
                qa_chain = RetrievalQA(
                    combine_documents_chain=combine_chain,
                    retriever=retriever,
                    return_source_documents=True
                )
                
                qa_chains.append(qa_chain)
                print(f"✅ Created QA chain {i}")
                
            except Exception as e:
                print(f"❌ Failed to create QA chain {i}: {e}")
        
        if not qa_chains:
            raise Exception("Failed to create any QA chains")
        
        print("✅ Multi-LLM chains ready!")
        return qa_chains
    
    def initialize(self):
        """Initialize the complete system"""
        try:
            print("🚀 Initializing ROCm Assistant...")
            retriever = self.setup_knowledge_base()
            self.qa_chains = self.setup_llm_chain(retriever)
            self.setup_complete = True
            print("🎉 ROCm Assistant ready!")
            return True
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def get_response(self, question):
        """Get response from multiple LLM instances and return the best one"""
        if not self.setup_complete or not self.qa_chains:
            return "❌ Assistant not properly initialized. Please restart."
        
        try:
            start_time = time.time()
            responses = []
            metrics = []
            
            # Get responses from all available instances
            for i, qa_chain in enumerate(self.qa_chains, 1):
                instance_start = time.time()
                try:
                    result = qa_chain(question)
                    instance_end = time.time()
                    
                    instance_time = instance_end - instance_start
                    response_length = len(result["result"])
                    
                    responses.append({
                        "instance": i,
                        "result": result["result"],
                        "sources": result.get("source_documents", []),
                        "time": instance_time,
                        "length": response_length
                    })
                    
                    metrics.append({
                        "instance": i,
                        "time": instance_time,
                        "status": "✅ Success",
                        "length": response_length
                    })
                    
                except Exception as e:
                    metrics.append({
                        "instance": i,
                        "time": 0,
                        "status": f"❌ Error: {str(e)[:30]}...",
                        "length": 0
                    })
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if not responses:
                return "❌ All LLM instances failed to respond. Please check Ollama servers."
            
            # Select the best response (longest and most detailed)
            best_response = max(responses, key=lambda x: x["length"])
            
            # Format the response
            response = best_response["result"]
            
            # Add sources
            if best_response["sources"]:
                response += "\n\n**📚 Sources:**\n"
                for i, doc in enumerate(best_response["sources"][:3], 1):
                    source_url = doc.metadata.get("source", "Unknown")
                    response += f"{i}. {source_url}\n"
            
            # Add performance metrics
            response += f"\n\n**⚡ Performance Metrics:**\n"
            response += f"**Best Response:** Instance {best_response['instance']} ({best_response['time']:.1f}s)\n"
            response += f"**Total Response Time:** {total_time:.1f}s\n"
            response += f"**Active Instances:** {len(responses)}/{len(self.qa_chains)}\n"
            
            # Add instance status
            response += f"\n**🤖 Instance Status:**\n"
            for metric in metrics:
                response += f"Instance {metric['instance']}: {metric['status']} ({metric['time']:.1f}s)\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

def create_interface():
    """Create the Gradio interface"""
    
    # Initialize assistant
    assistant = ROCmAssistant()
    
    # Custom CSS for beautiful interface
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .chat-container {
        border-radius: 15px;
        border: 2px solid #e1e5e9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-panel {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .status-panel {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-top: 1rem;
    }
    
    .example-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="ROCm Installation Assistant") as demo:
        
        # Header
        gr.HTML("""
        <div class="header-container">
            <div class="header-title">🚀 ROCm Installation Assistant</div>
            <div class="header-subtitle">Multi-LLM Expert guidance for AMD ROCm installation, configuration & troubleshooting</div>
        </div>
        """)
        
        # Status indicator
        status_display = gr.HTML("")
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<div class="info-panel"><h3>💬 Ask Your ROCm Questions</h3><p>Get expert help with installation, configuration, and troubleshooting</p></div>')
                
                chatbot = gr.Chatbot(
                    label="ROCm Expert Assistant",
                    height=500,
                    show_label=False,
                    elem_classes=["chat-container"],
                    type="messages"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Example: How do I install ROCm on Ubuntu 22.04?",
                        container=False,
                        scale=4,
                        lines=2
                    )
                    submit_btn = gr.Button("Ask Expert 🔍", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary", scale=1)
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="info-panel">
                    <h3>🎯 What I Can Help With</h3>
                    <ul>
                        <li><strong>Installation</strong> - Step-by-step guides</li>
                        <li><strong>Prerequisites</strong> - System requirements</li>
                        <li><strong>Configuration</strong> - Setup optimization</li>
                        <li><strong>Troubleshooting</strong> - Error resolution</li>
                        <li><strong>Verification</strong> - Testing your setup</li>
                        <li><strong>GPU Support</strong> - Compatibility check</li>
                    </ul>
                </div>
                """)
                
                gr.HTML("""
                <div class="status-panel">
                    <h4>💡 Pro Tips</h4>
                    <p><strong>Be specific:</strong> Include your OS version, GPU model, and exact error messages</p>
                    <p><strong>Follow steps:</strong> Execute commands in order and check each step</p>
                    <p><strong>Verify setup:</strong> Always test your installation</p>
                </div>
                """)
        
        # Example questions
        gr.HTML('<h3 style="text-align: center; margin: 2rem 0 1rem 0;">📋 Common Questions</h3>')
        
        examples = [
            "How do I install ROCm on Ubuntu 22.04?",
            "What are the system requirements for ROCm?",
            "My ROCm installation failed with package conflicts, how do I fix it?",
            "How do I verify that ROCm is working correctly?",
            "What GPUs are supported by ROCm?",
            "How do I uninstall and reinstall ROCm?",
            "ROCm is installed but PyTorch can't see my GPU, what's wrong?",
            "How do I update ROCm to the latest version?"
        ]
        
        gr.Examples(
            examples=examples,
            inputs=msg,
            examples_per_page=4
        )
        
        # Initialize the assistant
        def initialize_assistant():
            success = assistant.initialize()
            if success:
                num_instances = len(assistant.qa_chains)
                return f"""
                <div class="status-panel">
                    <h4>✅ Assistant Ready!</h4>
                    <p>Knowledge base loaded with latest ROCm documentation</p>
                    <p>Connected to {num_instances} Ollama LLM instances - Multi-processing ready!</p>
                    <p><strong>Performance:</strong> Parallel processing for faster responses</p>
                </div>
                """
            else:
                return """
                <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); padding: 1rem; border-radius: 10px; color: white; margin-top: 1rem;">
                    <h4>❌ Initialization Failed</h4>
                    <p>Please ensure Ollama instances are running:</p>
                    <p><code>./start_ollama.sh</code> or manually:</p>
                    <p><code>OLLAMA_HOST=0.0.0.0:11438 ollama serve</code></p>
                    <p><code>OLLAMA_HOST=0.0.0.0:11439 ollama serve</code></p>
                    <p><code>OLLAMA_HOST=0.0.0.0:11440 ollama serve</code></p>
                    <p><code>OLLAMA_HOST=0.0.0.0:11441 ollama serve</code></p>
                    <p>And model available: <code>ollama pull llama3.1:8b</code></p>
                </div>
                """
        
        def chat_response(message, history):
            if not message.strip():
                return history, ""
            
            if not assistant.setup_complete:
                error_msg = "❌ Assistant not initialized. Please check that Ollama is running and restart."
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": error_msg})
                return history, ""
            
            # Get response
            response = assistant.get_response(message)
            
            # Update chat history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            return history, ""
        
        def clear_chat():
            return [], ""
        
        # Event handlers
        submit_btn.click(chat_response, [msg, chatbot], [chatbot, msg])
        msg.submit(chat_response, [msg, chatbot], [chatbot, msg])
        clear_btn.click(clear_chat, outputs=[chatbot])
        
        # Initialize on load
        demo.load(initialize_assistant, outputs=[status_display])
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 2rem; margin-top: 2rem; border-top: 1px solid #dee2e6;">
            <p><strong>ROCm Installation Assistant</strong> - Powered by AMD ROCm Official Documentation</p>
            <p style="font-size: 0.9em; color: #6c757d;">Multi-LLM architecture with 4 parallel Llama 3.1 8B instances for optimal performance</p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting ROCm Installation Assistant...")
    print("🌐 Web interface will be available at: http://0.0.0.0:7862")
    
    demo = create_interface()
    try:
        demo.launch(
            share=True,
            server_name="0.0.0.0", 
            server_port=7862
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting interface: {e}")
        print("Please ensure all dependencies are installed and Ollama is running")
