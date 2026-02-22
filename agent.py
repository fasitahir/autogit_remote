"""AI Git Agent - LangChain Agent with Hugging Face Model"""
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tools import (
    git_status,
    git_add,
    git_commit,
    git_init,
    git_branch_rename,
    get_branch_info,
    diagnose_git_config,
    get_remote_url,
    git_remote_add,
    git_push,
    generate_version_documentation,
    resolve_conflicts,
    get_merge_conflicts,
    validate_git_repository,
    git_reinitialize
)


class AIGitAgent:
    def __init__(self, hf_token: str, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.hf_token = hf_token
        self.model_id = model_id
        self.tools = self._load_tools()
        self.tools_dict = {tool.name: tool for tool in self.tools}
        
        # Create LLM with Hugging Face Endpoint
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            huggingfacehub_api_token=hf_token,
            temperature=0.1,
            max_new_tokens=800,
            top_k=50,
        )
        
        # Create chat model wrapper
        base_llm = ChatHuggingFace(llm=llm, verbose=True)
        
        # Bind tools to the model
        self.llm = base_llm.bind_tools(self.tools)
        self.chat_history = []
        
    def _load_tools(self) -> list:
        """Load all available LangChain tools"""
        return [
            git_status,
            git_add,
            git_commit,
            git_init,
            git_branch_rename,
            get_branch_info,
            diagnose_git_config,
            get_remote_url,
            git_remote_add,
            git_push,
            generate_version_documentation,
            resolve_conflicts,
            get_merge_conflicts,
            validate_git_repository,
            git_reinitialize
        ]
    
    def _get_system_message(self) -> str:
        """Get system prompt"""
        return """You are a friendly and helpful Git automation assistant.

IMPORTANT: Always show the ACTUAL output from tools. Never say something succeeded unless you see a success message from the tool.

⚠️ REPOSITORY CONTEXT NOTE:
The repository metadata you see (owner: user, repo: repo) is just an EXAMPLE placeholder.
NEVER use "https://github.com/user/repo.git" or auto-generate URLs from this metadata.
ALWAYS ask the user for their actual GitHub repository URL when needed.

GUIDELINES:
• Respond naturally to greetings and casual conversation
• For Git-related requests, use the appropriate tools to help
• Only use tools when the user asks for Git operations
• ALWAYS report what the tools actually returned - don't make assumptions
• If a tool shows an error, report that error to the user
• Be conversational and friendly
• When git_remote_add is needed, ASK the user for their actual repository URL first

WHEN TO USE TOOLS:

0. BEFORE ANY GIT OPERATIONS (CRITICAL):
   • ALWAYS call validate_git_repository FIRST before doing any push, commit, or remote operations
   • This checks if Git is initialized in the correct folder
   • If validation shows issues, inform user and ask what they want to do
   • If user wants to reinitialize, call git_reinitialize
   • Only proceed with other operations after validation passes

1. Repository Status:
   • "check status" / "status" / "what changed" → use git_status
   
2. Branch Info:
   • "what branch" / "current branch" → use get_branch_info
   
3. Configuration:
   • "diagnose" / "check config" → use diagnose_git_config
   • "get remote" / "remote url" → use get_remote_url
   
4. Commit ONLY:
   • "commit" / "save changes" / "commit with message X" → 
     a) Call git_add to stage files
     b) Call git_commit with the message
     c) Report actual results
     DO NOT PUSH unless explicitly asked!
   
5. Push Code:
   • "push" / "push my code" / "upload to github" → You MUST do ALL these steps IN ORDER:
     a) Call validate_git_repository FIRST (CRITICAL - detects wrong repo configuration)
     b) If validation fails or shows parent repo, stop and inform user
     c) If user wants to reinitialize, call git_reinitialize
     d) Check git_status
     e) Call git_add to stage changes
     f) Call generate_version_documentation to create detailed PDF documentation
     g) Call git_commit with a commit message
     h) Call git_push
     i) Report the actual results from each step
   
6. Initialize/Reinitialize:
   • "init" / "create repo" → use git_init
   • "reinitialize" / "reinit" / "fresh start" → use git_reinitialize
   • "validate" / "check git" / "check repo" → use validate_git_repository

7. Generate Documentation:
   • "generate docs" / "create documentation" / "document changes" / "document my code" →
     Call generate_version_documentation ONCE to create a detailed PDF report
     It automatically detects current changes (staged or unstaged)
     IMPORTANT: Only call this tool ONE TIME, then report the result

8. Merge Conflict Resolution:
   • "resolve conflicts" / "fix merge conflicts" / "merge conflict" →
     a) First call get_merge_conflict_info to analyze conflicts
     b) Then call resolve_merge_conflicts with strategy='ai' for intelligent resolution
     c) Report results and next steps
   • "check conflicts" / "show conflicts" → use get_merge_conflict_info
   • For manual strategy: resolve_merge_conflicts(strategy='ours'|'theirs'|'both')
   
WHEN NOT TO USE TOOLS:
• Greetings: "hi", "hello", "hey" → Just greet back warmly
• Questions: "how are you", "what can you do" → Explain your capabilities
• Thanks: "thank you", "thanks" → Acknowledge politely
• Casual chat → Respond naturally without using tools

CRITICAL: Base your response ONLY on actual tool outputs. If you don't call a tool, don't claim it succeeded."""

    def run(self, user_command: str):
        """Main method to process commands using Hugging Face model with tool calling"""
        print(f"\n💻 User: {user_command}\n")
        print("🤖 AI Agent: Processing your request...\n")
        
        try:
            # Add system message and user message
            messages = [
                {"role": "system", "content": self._get_system_message()},
                *self.chat_history,
                HumanMessage(content=user_command)
            ]
            
            max_iterations = 8
            iteration = 0
            tool_call_tracker = {}  # Track which tools have been called
            
            while iteration < max_iterations:
                iteration += 1
                
                # Get response from LLM
                response = self.llm.invoke(messages)
                messages.append(response)
                
                # Check if there are tool calls
                if not response.tool_calls:
                    # No more tool calls, print final response
                    if response.content:
                        print(f"\n✅ {response.content}\n")
                    break
                
                # Execute all tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("args", {})
                    
                    # Prevent duplicate calls to generate_version_documentation
                    if tool_name == "generate_version_documentation":
                        if tool_name in tool_call_tracker:
                            # Silently skip duplicate - don't show message to user
                            tool_message = ToolMessage(
                                content="Documentation already generated.",
                                tool_call_id=tool_call["id"]
                            )
                            messages.append(tool_message)
                            continue
                        else:
                            tool_call_tracker[tool_name] = True
                    
                    print(f"🔧 Calling: {tool_name}")
                    
                    if tool_name in self.tools_dict:
                        # Execute the tool
                        result = self.tools_dict[tool_name].invoke(tool_args)
                        
                        # Display result
                        if isinstance(result, dict):
                            msg = result.get('message', result.get('output', str(result)))
                            print(f"{msg}\n")
                        else:
                            print(f"{result}\n")
                        
                        # Add tool response to messages
                        tool_message = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                        messages.append(tool_message)
                    else:
                        error_msg = f"❌ Tool {tool_name} not found"
                        print(f"{error_msg}\n")
                        tool_message = ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call["id"]
                        )
                        messages.append(tool_message)
            
            # Save chat history (excluding system message)
            self.chat_history = messages[1:]
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
        
        print("✨ Done!\n")
