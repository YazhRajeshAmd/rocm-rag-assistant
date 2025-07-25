# ROCm RAG Assistant

🚀 **Multi-LLM ROCm Installation & Troubleshooting Assistant with RAG Architecture**

A sophisticated Retrieval-Augmented Generation (RAG) system specifically designed for AMD ROCm installation, configuration, and troubleshooting. Features multi-LLM architecture with parallel processing for optimal performance.

## 🎯 Features

### 🧠 Multi-LLM Architecture
- **4 Parallel Ollama Instances**: Load-balanced across multiple GPU instances
- **Smart Response Selection**: Chooses the best response based on quality metrics
- **Performance Monitoring**: Real-time metrics and instance status tracking
- **Graceful Fallbacks**: Continues operation even if some instances fail

### 📚 Knowledge Base
- **ROCm Documentation Scraping**: Automatically fetches latest AMD ROCm docs
- **Vector Database**: ChromaDB with HuggingFace embeddings
- **Focused Content**: Installation, troubleshooting, and configuration guides
- **Smart Retrieval**: MMR (Maximal Marginal Relevance) for diverse, relevant results

### 🎨 Modern Interface
- **Beautiful Gradio UI**: Professional gradient design with responsive layout
- **Interactive Chat**: Real-time conversation with expert-level responses
- **Example Prompts**: Common ROCm questions for quick access
- **Performance Metrics**: Live display of LLM instance performance

## 🛠️ Technology Stack

- **Python 3.8+**
- **LangChain**: RAG orchestration and document processing
- **Ollama**: Multi-instance LLM serving (Llama 3.1 8B)
- **ChromaDB**: Vector database for semantic search
- **HuggingFace**: Sentence transformers for embeddings
- **BeautifulSoup**: Web scraping for documentation
- **Gradio**: Modern web interface

## 🚀 Quick Start

### Prerequisites

1. **AMD GPU with ROCm Support**
2. **Ollama Installation**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.1:8b
   ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rocm-rag-assistant.git
   cd rocm-rag-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start multiple Ollama instances:
   ```bash
   cd gpu1
   chmod +x start_ollama.sh
   ./start_ollama.sh
   ```

4. Run the enhanced assistant:
   ```bash
   python rocm_assistant_enhanced.py
   ```

5. Access the web interface at `http://localhost:7863`

## 📊 Architecture Overview

### Multi-Instance Setup
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ollama :11438 │    │   Ollama :11439 │    │   Ollama :11440 │    │   Ollama :11441 │
│  Llama 3.1 8B   │    │  Llama 3.1 8B   │    │  Llama 3.1 8B   │    │  Llama 3.1 8B   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         └───────────────────────┼───────────────────────┼───────────────────────┘
                                 │                       │
                    ┌─────────────────────────────────────────────────┐
                    │              RAG Assistant                       │
                    │          Smart Load Balancer                    │
                    └─────────────────────────────────────────────────┘
                                         │
                    ┌─────────────────────────────────────────────────┐
                    │              Vector Database                     │
                    │         (ChromaDB + Embeddings)                 │
                    └─────────────────────────────────────────────────┘
```

### RAG Pipeline
1. **Document Ingestion**: Scrapes latest ROCm documentation
2. **Text Chunking**: Splits documents with optimal overlap
3. **Embedding Generation**: Creates semantic vectors using HuggingFace
4. **Vector Storage**: Stores in ChromaDB with metadata
5. **Query Processing**: Retrieves relevant context using MMR
6. **Multi-LLM Generation**: Parallel response generation
7. **Response Selection**: Chooses best response based on quality metrics

## 🎯 Use Cases

### Installation Support
- **Fresh ROCm Installation**: Step-by-step guides for different Linux distributions
- **Package Management**: Repository setup, dependency resolution
- **Version Compatibility**: Matching ROCm versions with GPU architectures

### Configuration & Optimization
- **Environment Variables**: HIP_VISIBLE_DEVICES, ROCM_PATH setup
- **Performance Tuning**: Memory allocation, compute optimization
- **Multi-GPU Setup**: Configuring multiple AMD GPUs

### Troubleshooting
- **Installation Errors**: Package conflicts, dependency issues
- **Runtime Problems**: GPU detection, memory errors
- **Performance Issues**: Optimization recommendations

## 📁 Project Structure

```
rag_model/
├── gpu1/
│   ├── rocm_assistant_enhanced.py    # Enhanced multi-LLM assistant
│   ├── gpu1.py                       # Original multi-panel version
│   ├── start_ollama.sh              # Multi-instance startup script
│   ├── rocm_kb/                     # Vector database storage
│   └── logs/                        # Ollama instance logs
├── gpu4/
│   ├── gpu4.py                      # GPU4-specific version
│   ├── gpu4_metrics.py              # Performance monitoring
│   └── start_ollama_gpu4.sh         # GPU4 startup script
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## 🔧 Configuration

### Multi-Instance Setup
```bash
# Edit start_ollama.sh to configure:
GPU_TO_USE=5                    # AMD GPU index
ports=(11438 11439 11440 11441) # Ollama ports
```

### RAG Parameters
```python
# Vector database configuration
chunk_size=1000                 # Document chunk size
chunk_overlap=200              # Overlap between chunks
search_k=5                     # Retrieved documents per query
```

### LLM Settings
```python
# Ollama configuration
model="llama3.1:8b"            # Model name
temperature=0.1                # Response randomness (low for technical accuracy)
```

## 🚀 Advanced Features

### Performance Monitoring
- **Response Time Tracking**: Per-instance timing metrics
- **Quality Assessment**: Response length and completeness scoring
- **Load Balancing**: Automatic distribution across healthy instances
- **Health Checks**: Continuous monitoring of instance availability

### Smart Response Selection
- **Quality Metrics**: Length, completeness, and relevance scoring
- **Performance Weighting**: Faster responses get priority in ties
- **Fallback Handling**: Graceful degradation when instances fail
- **Source Attribution**: Tracks which documentation sources were used

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/rocm-enhancement`)
3. Commit your changes (`git commit -m 'Add ROCm enhancement'`)
4. Push to the branch (`git push origin feature/rocm-enhancement`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [Financial Stock Analysis](https://github.com/yourusername/fsi-stock-analysis)
- [MRI Analysis Tool](https://github.com/yourusername/mri-analysis-tool)

## 📚 Documentation Sources

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html)
- [System Requirements](https://rocm.docs.amd.com/en/latest/deploy/linux/prerequisites.html)

## ⚡ Performance

- **Multi-LLM Processing**: 4x parallel processing capability
- **Response Time**: < 5 seconds for most queries
- **Throughput**: Handles multiple concurrent users
- **Accuracy**: Expert-level ROCm knowledge with source attribution

---

**Built with ❤️ for the AMD ROCm community**
