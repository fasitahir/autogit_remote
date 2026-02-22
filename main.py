"""Main execution script for AI Git Agent"""
import os
from dotenv import load_dotenv
from agent import AIGitAgent

# Load environment variables
load_dotenv()


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("🤖 AI Git Agent - Powered by Hugging Face LLM")
    print("   🔍 Smart Error Detection & Solutions")
    print("="*60)
    
    # Load API key from .env
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    model_id = os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
    
    if not hf_token:
        print("\n❌ HUGGINGFACE_TOKEN not found in .env file!")
        print("📝 Create a .env file with: HUGGINGFACE_TOKEN=your_token_here")
        print("🔑 Get your token from: https://huggingface.co/settings/tokens")
        print("\n💡 Optional: Set HF_MODEL_ID=model_name to use a specific model")
        print("   Default: meta-llama/Meta-Llama-3-8B-Instruct")
        exit()
    
    print("✅ Hugging Face token loaded from .env")
    print(f"📦 Using model: {model_id}")
    print("\n📚 Available Commands:")
    print("   • 'push my code' - Push code to GitHub with smart error handling")
    print("   • 'diagnose' - Check Git configuration for issues")
    print("   • 'status' - Check repository status")
    print("   • 'generate docs' - Create detailed PDF documentation of changes")
    print("   • Any Git-related request in natural language!")
    
    agent = AIGitAgent(hf_token=hf_token, model_id=model_id)
    
    while True:
        print("\n" + "-"*60)
        command = input("\n💬 You: ").strip()
        
        if command.lower() in ['exit', 'quit', 'bye']:
            print("\n👋 Goodbye!")
            break
        
        if command:
            try:
                agent.run(command)
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
