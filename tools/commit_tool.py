import subprocess
import os
import re
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables
load_dotenv()

# Token limits for safety
MAX_TOKENS_PER_CHUNK = 2000  # Characters per chunk (roughly 500 tokens)
MAX_TOTAL_CONTEXT = 6000     # Max total context to send to LLM


def _should_ignore_file(filepath: str) -> bool:
    """Check if file should be ignored for commit message generation."""
    ignore_patterns = [
        'pycache', '.pyc', '.pyo', '.pyd', '.so', '.dll', '.class', '.o',
        'node_modules', '.git/', 'package-lock.json', 'yarn.lock', '.DS_Store',
        '.lock', 'dist/', 'build/', '_pycache_', '.egg-info'
    ]
    return any(pattern in filepath for pattern in ignore_patterns)


def _extract_function_class_names(code_line: str) -> str:
    """Extract function or class names from code for better context."""
    # Python function/class
    if 'def ' in code_line:
        match = re.search(r'def\s+(\w+)', code_line)
        if match:
            return f"function '{match.group(1)}'"
    if 'class ' in code_line:
        match = re.search(r'class\s+(\w+)', code_line)
        if match:
            return f"class '{match.group(1)}'"
    
    # JavaScript/TypeScript function
    if 'function ' in code_line:
        match = re.search(r'function\s+(\w+)', code_line)
        if match:
            return f"function '{match.group(1)}'"
    if 'const ' in code_line and ('=>' in code_line or '= function' in code_line):
        match = re.search(r'const\s+(\w+)', code_line)
        if match:
            return f"function '{match.group(1)}'"
    
    return None


def _analyze_change_intent(snippets: list, adds: int, dels: int, filepath: str) -> str:
    """Analyze what kind of change this represents based on code patterns."""
    snippet_text = ' '.join(snippets).lower()
    filename = os.path.basename(filepath).lower()
    
    # Check for specific patterns
    if adds > 0 and dels == 0:
        if adds > 20:
            return "NEW FILE"
        return "new additions"
    
    if dels > 0 and adds == 0:
        return "deletions"
    
    # Analyze content
    if 'import' in snippet_text or 'require' in snippet_text:
        return "dependency changes"
    
    if 'def ' in snippet_text or 'function ' in snippet_text or 'class ' in snippet_text:
        if adds > dels * 2:
            return "added functions/classes"
        elif dels > adds * 2:
            return "removed functions"
        else:
            return "refactored functions"
    
    if 'test' in filename or 'spec' in filename:
        return "test updates"
    
    if 'readme' in filename or '.md' in filename:
        return "documentation"
    
    if any(word in snippet_text for word in ['fix', 'bug', 'error', 'issue']):
        return "bug fix"
    
    if 'return' in snippet_text or 'if ' in snippet_text:
        return "logic changes"
    
    # Default based on ratio
    if adds > dels * 2:
        return "major additions"
    elif dels > adds * 2:
        return "major deletions"
    else:
        return "modifications"


def _parse_git_diff_summary(diff_output: str) -> dict:
    """Parse git diff with intelligent extraction."""
    files_changed = []
    current_file = None
    additions = 0
    deletions = 0
    added_snippets = []
    removed_snippets = []
    
    lines = diff_output.split('\n')
    
    for line in lines:
        if line.startswith('diff --git'):
            # Save previous file
            if current_file and not _should_ignore_file(current_file):
                change_intent = _analyze_change_intent(
                    added_snippets + removed_snippets, 
                    additions, 
                    deletions,
                    current_file
                )
                
                files_changed.append({
                    'file': current_file,
                    'basename': os.path.basename(current_file),
                    'additions': additions,
                    'deletions': deletions,
                    'total_changes': additions + deletions,
                    'added_snippets': added_snippets[:6],
                    'removed_snippets': removed_snippets[:3],
                    'change_intent': change_intent,
                    'file_type': _get_file_type(current_file)
                })
            
            # New file
            match = re.search(r'b/(.+)$', line)
            if match:
                current_file = match.group(1)
                additions = 0
                deletions = 0
                added_snippets = []
                removed_snippets = []
        
        elif current_file and not _should_ignore_file(current_file):
            # Added lines
            if line.startswith('+') and not line.startswith('+++'):
                additions += 1
                stripped = line[1:].strip()
                
                if stripped and len(stripped) > 5 and not stripped.startswith('#'):
                    # Extract function/class names
                    func_class_name = _extract_function_class_names(stripped)
                    if func_class_name:
                        added_snippets.append(f"➕ {func_class_name}")
                    elif any(kw in stripped for kw in ['import ', 'from ', 'require']):
                        # Track imports briefly
                        match = re.search(r'import\s+(\w+)|from\s+([\w.]+)', stripped)
                        if match:
                            module = match.group(1) or match.group(2)
                            added_snippets.append(f"Import {module}")
                    elif len(stripped) < 80 and any(kw in stripped for kw in ['=', 'return ', 'if ', 'else', 'raise ', 'throw ']):
                        added_snippets.append(stripped[:70])
            
            # Removed lines
            elif line.startswith('-') and not line.startswith('---'):
                deletions += 1
                stripped = line[1:].strip()
                
                if stripped and len(stripped) > 5:
                    func_class_name = _extract_function_class_names(stripped)
                    if func_class_name:
                        removed_snippets.append(f"➖ {func_class_name}")
    
    # Save last file
    if current_file and not _should_ignore_file(current_file):
        change_intent = _analyze_change_intent(
            added_snippets + removed_snippets, 
            additions, 
            deletions,
            current_file
        )
        
        files_changed.append({
            'file': current_file,
            'basename': os.path.basename(current_file),
            'additions': additions,
            'deletions': deletions,
            'total_changes': additions + deletions,
            'added_snippets': added_snippets[:6],
            'removed_snippets': removed_snippets[:3],
            'change_intent': change_intent,
            'file_type': _get_file_type(current_file)
        })
    
    return {
        'total_files': len(files_changed),
        'files': files_changed,
        'total_additions': sum(f['additions'] for f in files_changed),
        'total_deletions': sum(f['deletions'] for f in files_changed)
    }


def _get_file_type(filepath: str) -> str:
    """Determine file type/category."""
    filepath_lower = filepath.lower()
    
    if '.py' in filepath_lower:
        return 'python'
    elif any(ext in filepath_lower for ext in ['.js', '.jsx', '.ts', '.tsx']):
        return 'javascript'
    elif '.java' in filepath_lower:
        return 'java'
    elif any(ext in filepath_lower for ext in ['.md', '.txt', '.rst']):
        return 'docs'
    elif any(ext in filepath_lower for ext in ['.json', '.yaml', '.yml', '.toml', '.ini']):
        return 'config'
    elif any(ext in filepath_lower for ext in ['.html', '.css', '.scss', '.sass']):
        return 'frontend'
    elif 'test' in filepath_lower or 'spec' in filepath_lower:
        return 'test'
    else:
        return 'other'


def _prioritize_files(files: list) -> dict:
    """Categorize files by importance for intelligent processing."""
    critical = []    # >100 changes
    important = []   # 20-100 changes
    moderate = []    # 5-20 changes
    minor = []       # <5 changes
    
    for file_info in files:
        total = file_info['total_changes']
        if total > 100:
            critical.append(file_info)
        elif total >= 20:
            important.append(file_info)
        elif total >= 5:
            moderate.append(file_info)
        else:
            minor.append(file_info)
    
    return {
        'critical': critical,
        'important': important,
        'moderate': moderate,
        'minor': minor
    }


def _create_file_summary(file_info: dict, detail_level: str = 'full') -> str:
    """Create summary for a single file with variable detail level."""
    basename = file_info['basename']
    intent = file_info['change_intent']
    adds = file_info['additions']
    dels = file_info['deletions']
    
    if detail_level == 'minimal':
        # Just filename and stats
        return f"• {basename}: {intent} (+{adds}/-{dels})"
    
    parts = [f"• {basename}: {intent} (+{adds}/-{dels})"]
    
    if detail_level == 'full':
        # Include snippets
        added = file_info.get('added_snippets', [])
        removed = file_info.get('removed_snippets', [])
        
        if added:
            parts.append("  Changes:")
            for snippet in added[:3]:  # Max 3 snippets
                parts.append(f"    {snippet}")
        
        if removed and detail_level == 'full':
            for snippet in removed[:1]:  # Max 1 removed snippet
                parts.append(f"    {snippet}")
    
    return '\n'.join(parts)


def _create_chunked_summary(diff_summary: dict) -> str:
    """Create intelligent summary that respects token limits."""
    prioritized = _prioritize_files(diff_summary['files'])
    
    summary_parts = []
    current_size = 0
    
    # Header (always include)
    header = f"📊 CHANGES: {diff_summary['total_files']} files | +{diff_summary['total_additions']} -{diff_summary['total_deletions']} lines\n"
    summary_parts.append(header)
    current_size += len(header)
    
    # Process critical files first (full detail)
    if prioritized['critical']:
        summary_parts.append("\n🔥 CRITICAL CHANGES (>30 lines):")
        for file_info in prioritized['critical']:
            file_summary = _create_file_summary(file_info, 'full')
            if current_size + len(file_summary) < MAX_TOTAL_CONTEXT:
                summary_parts.append(file_summary)
                current_size += len(file_summary)
            else:
                # Token limit reached, use minimal
                file_summary = _create_file_summary(file_info, 'minimal')
                summary_parts.append(file_summary)
                current_size += len(file_summary)
    
    # Important files (medium detail)
    if prioritized['important']:
        summary_parts.append("\n📌 IMPORTANT CHANGES (20-100 lines):")
        for file_info in prioritized['important'][:10]:  # Max 10 important files
            detail = 'full' if current_size < MAX_TOTAL_CONTEXT * 0.6 else 'minimal'
            file_summary = _create_file_summary(file_info, detail)
            if current_size + len(file_summary) < MAX_TOTAL_CONTEXT:
                summary_parts.append(file_summary)
                current_size += len(file_summary)
    
    # Moderate files (minimal detail)
    if prioritized['moderate']:
        summary_parts.append(f"\n📝 MODERATE CHANGES: {len(prioritized['moderate'])} files (5-20 lines each)")
        for file_info in prioritized['moderate'][:5]:  # Show first 5
            if current_size < MAX_TOTAL_CONTEXT * 0.8:
                file_summary = _create_file_summary(file_info, 'minimal')
                summary_parts.append(file_summary)
                current_size += len(file_summary)
    
    # Minor files (just count)
    if prioritized['minor']:
        summary_parts.append(f"\n✨ MINOR CHANGES: {len(prioritized['minor'])} files (<5 lines each)")
        # List just the names
        if current_size < MAX_TOTAL_CONTEXT * 0.9:
            names = [f['basename'] for f in prioritized['minor'][:10]]
            summary_parts.append(f"  Files: {', '.join(names)}")
    
    result = '\n'.join(summary_parts)
    
    # Final safety check
    if len(result) > MAX_TOTAL_CONTEXT:
        result = result[:MAX_TOTAL_CONTEXT] + "\n\n⚠ [Summary truncated to fit context limit]"
    
    return result


def _generate_commit_with_llm(structured_context: str, total_files: int) -> str:
    """Generate commit message using LLM with the structured context."""
    try:
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct"),
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN"),
            temperature=0.2,
            max_new_tokens=100,
            top_k=50,
        )
        llm = ChatHuggingFace(llm=llm_endpoint)
        
        prompt = f"""You are a Git commit expert. Analyze ALL changes below and create commit message.

{structured_context}

CRITICAL RULES:
1. READ EVERYTHING - understand the MAIN PURPOSE across all {total_files} files
2. Identify the PRIMARY FEATURE/FIX that ties everything together
3. Format: <type>(<scope>): <specific description>
   Types: feat|fix|refactor|docs|style|test|chore|perf
   Scope: main module/feature affected
4. Be SPECIFIC - mention actual function/feature names when possible
5. Max 70 characters, imperative mood
6. ONE to TWO LINES ONLY - no explanations but intelligently concise

Examples:
✅ feat(git-tool): Add intelligent chunked diff processing
✅ refactor(commit): Extract file prioritization and summarization
✅ fix(parser): Resolve token limit issues in large diffs
✅ feat(auth): Implement JWT authentication with refresh tokens

❌ BAD (never do):
- chore: update files
- feat: improve code
- refactor: enhance functionality

Your commit message (one line):"""
        
        response = llm.invoke(prompt)
        
        if response.content:
            message = response.content.strip()
            # Clean up
            message = message.split('\n')[0].strip('"').strip("'").strip()
            
            # Validate quality
            generic_words = ['enhance', 'improve', 'update files', 'modify code', 'change', 'update code']
            is_generic = any(word in message.lower() for word in generic_words)
            is_too_short = len(message) < 20
            
            if is_generic and is_too_short:
                # Return None to trigger fallback
                return None
            
            return message
        
        return None
        
    except Exception as e:
        print(f"⚠ LLM error: {str(e)}")
        return None


def _create_fallback_message(diff_summary: dict) -> str:
    """Create a reasonable fallback commit message."""
    files = diff_summary['files']
    total_files = diff_summary['total_files']
    
    if total_files == 1:
        file_info = files[0]
        intent = file_info['change_intent']
        basename = file_info['basename'].replace('.py', '').replace('.js', '')
        return f"refactor({basename}): {intent}"
    
    # Find most common file type
    file_types = [f['file_type'] for f in files]
    most_common_type = max(set(file_types), key=file_types.count)
    
    # Find most common intent
    intents = [f['change_intent'] for f in files]
    most_common_intent = max(set(intents), key=intents.count)
    
    if most_common_type == 'docs':
        return f"docs: Update documentation across {total_files} files"
    elif most_common_type == 'test':
        return f"test: Update test suite ({total_files} files)"
    elif 'NEW FILE' in most_common_intent:
        return f"feat({most_common_type}): Add {total_files} new files"
    else:
        return f"refactor({most_common_type}): {most_common_intent} across {total_files} files"


@tool
def git_commit(message: str = "auto") -> dict:
    """Create a commit with the specified message. If message is 'auto', generates a meaningful commit message based on git diff.
    
    Args:
        message: The commit message to use. Use 'auto' to automatically generate a meaningful message based on changes.
    
    Returns:
        Dictionary with status and message about the commit operation.
    """
    if message.lower() == "auto":
        try:
            # Get staged changes first
            diff_result = subprocess.run(
                ['git', 'diff', '--cached'], 
                capture_output=True, 
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            # If no staged changes, check unstaged
            if not diff_result.stdout.strip():
                print("🔍 No staged changes, checking unstaged...")
                diff_result = subprocess.run(
                    ['git', 'diff'], 
                    capture_output=True, 
                    text=True,
                    encoding='utf-8',
                    errors='ignore'
                )
            
            git_diff_output = diff_result.stdout.strip()
            
            if not git_diff_output:
                return {
                    "status": "error", 
                    "message": "❌ No changes detected\n💡 Make changes or run 'git add <files>' first"
                }
            
            print(f"🔍 Analyzing {len(git_diff_output):,} characters of git diff...\n")
            
            # Parse diff with intelligent extraction
            diff_summary = _parse_git_diff_summary(git_diff_output)
            
            print(f"📊 Detected changes in {diff_summary['total_files']} file(s):")
            print(f"   Total: +{diff_summary['total_additions']} -{diff_summary['total_deletions']} lines\n")
            
            # Prioritize files
            prioritized = _prioritize_files(diff_summary['files'])
            print(f"🎯 Priority breakdown:")
            print(f"   🔥 Critical (>100 lines): {len(prioritized['critical'])} files")
            print(f"   📌 Important (20-100): {len(prioritized['important'])} files")
            print(f"   📝 Moderate (5-20): {len(prioritized['moderate'])} files")
            print(f"   ✨ Minor (<5): {len(prioritized['minor'])} files\n")
            
            # Create intelligent summary with token management
            structured_context = _create_chunked_summary(diff_summary)
            
            print(f"📦 Context size: {len(structured_context):,} chars (limit: {MAX_TOTAL_CONTEXT:,})")
            print(f"✅ Context fits within token limits!\n")
            
            # Show what LLM will see (first 400 chars)
            print("📝 Summary preview for LLM:")
            print("─" * 60)
            preview = structured_context[:400] + "..." if len(structured_context) > 400 else structured_context
            print(preview)
            print("─" * 60)
            print()
            
            # Generate commit message with LLM
            print("🤖 Generating commit message with LLM...")
            message = _generate_commit_with_llm(structured_context, diff_summary['total_files'])
            
            # Fallback if LLM fails or produces generic message
            if not message:
                print("⚠ LLM produced generic message, using intelligent fallback...")
                message = _create_fallback_message(diff_summary)
            
            print(f"✅ Generated commit: {message}\n")
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"❌ Git diff error: {str(e)}"
            }
    
    # Execute commit
    result = subprocess.run(
        ['git', 'commit', '-m', message], 
        capture_output=True, 
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    
    if result.returncode == 0:
        return {"status": "success", "message": f"✅ Committed: {message}"}
    
    error_msg = result.stderr.strip() or result.stdout.strip() or "Commit failed"
    
    solution = ""
    if "nothing to commit" in error_msg.lower():
        solution = "\n💡 No changes to commit. Stage files with 'git add <files>' first"
    elif "please tell me who you are" in error_msg.lower():
        solution = "\n💡 Configure git:\n   git config --global user.name \"Your Name\"\n   git config --global user.email \"email@example.com\""
    
    return {"status": "error", "message": f"❌ {error_msg}{solution}"}