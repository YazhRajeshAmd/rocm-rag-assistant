#!/usr/bin/env python3
"""
Simple diagnosis script for ROCm AI Assistant
"""

def check_dependencies():
    """Check if all required packages are installed"""
    missing = []
    try:
        import requests
        print("✅ requests")
    except ImportError:
        missing.append("requests")
        print("❌ requests")
    
    try:
        import gradio
        print("✅ gradio")
    except ImportError:
        missing.append("gradio")
        print("❌ gradio")
    
    try:
        import langchain
        print("✅ langchain")
    except ImportError:
        missing.append("langchain")
        print("❌ langchain")
    
    try:
        from langchain_community.llms import Ollama
        print("✅ langchain_community")
    except ImportError:
        missing.append("langchain_community")
        print("❌ langchain_community")
    
    try:
        from bs4 import BeautifulSoup
        print("✅ beautifulsoup4")
    except ImportError:
        missing.append("beautifulsoup4")
        print("❌ beautifulsoup4")
    
    return missing

def check_ollama():
    """Check Ollama instances"""
    import requests
    ports = [11434, 11435, 11436, 11437]
    working = []
    
    for port in ports:
        try:
            response = requests.get(f"http://localhost:{port}/api/tags", timeout=3)
            if response.status_code == 200:
                working.append(port)
                print(f"✅ Ollama on port {port}")
            else:
                print(f"❌ Ollama on port {port} - HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Ollama on port {port} - {e}")
    
    return working

def main():
    print("🔍 ROCm AI Assistant - Diagnosis")
    print("=" * 40)
    
    print("\n📦 Checking Python Dependencies:")
    missing = check_dependencies()
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with:")
        print(f"   pip install --break-system-packages {' '.join(missing)}")
        print("   or use a virtual environment")
    else:
        print("\n✅ All dependencies installed!")
    
    print("\n🤖 Checking Ollama Instances:")
    working = check_ollama()
    
    if not working:
        print("\n❌ No Ollama instances running!")
        print("Start with: ./start_ollama.sh")
    else:
        print(f"\n✅ {len(working)} Ollama instances ready!")
    
    if not missing and working:
        print("\n🎉 System is ready! Run: python3 gpu1.py")
    else:
        print("\n⚠️  System not ready. Fix the issues above first.")

if __name__ == "__main__":
    main()
