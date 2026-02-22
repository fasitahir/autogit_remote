"""Main execution script for AI Git Agent"""
import os
from dotenv import load_dotenv
from agent import AIGitAgent

# Load environment variables
load_dotenv()


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("🤖 AI Git Agent - Powered by Hugging Face LLM (Offline)")
    print("   🔍 Smart Error Detection & Solutions")
    print("="*60)
    
    # Load model ID from .env or use default
    model_id = os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
    
    print(f"\n📦 Using offline model: {model_id}")
    print("⚠️  Note: First run will download the model if not cached")
    
    print("\n📚 Available Commands:")
    print("   • 'push my code' - Push code to GitHub with smart error handling")
    print("   • 'diagnose' - Check Git configuration for issues")
    print("   • 'status' - Check repository status")
    print("   • 'generate docs' - Create detailed PDF documentation of changes")
    print("   • Any Git-related request in natural language!")
    
    agent = AIGitAgent(model_id=model_id)
    
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
