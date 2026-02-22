"""Advanced Documentation Tool - AI-Powered PDF Documentation Generation"""
import subprocess
import os
import re
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime

# Load environment variables
load_dotenv()

# Token limits for documentation generation
MAX_TOKENS_PER_FILE = 3000  # Characters per file chunk
MAX_TOTAL_CONTEXT = 12000   # Max total context to send to LLM per pass


def _should_ignore_file(filepath: str) -> bool:
    """Check if file should be ignored for documentation generation."""
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


def _parse_git_diff_for_documentation(diff_output: str) -> dict:
    """Parse git diff with intelligent extraction for documentation."""
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
                    'added_snippets': added_snippets[:10],  # More snippets for documentation
                    'removed_snippets': removed_snippets[:5],
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
                        match = re.search(r'import\s+(\w+)|from\s+([\w.]+)', stripped)
                        if match:
                            module = match.group(1) or match.group(2)
                            added_snippets.append(f"Import {module}")
                    elif len(stripped) < 100:
                        added_snippets.append(stripped[:90])
            
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
            'added_snippets': added_snippets[:10],
            'removed_snippets': removed_snippets[:5],
            'change_intent': change_intent,
            'file_type': _get_file_type(current_file)
        })
    
    return {
        'total_files': len(files_changed),
        'files': files_changed,
        'total_additions': sum(f['additions'] for f in files_changed),
        'total_deletions': sum(f['deletions'] for f in files_changed)
    }


def _prioritize_files_for_documentation(files: list) -> dict:
    """Categorize files by importance for intelligent processing."""
    critical = []    # >100 changes - highest detail
    important = []   # 20-100 changes - medium detail
    moderate = []    # 5-20 changes - basic detail
    minor = []       # <5 changes - minimal detail
    
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


def _create_file_detail(file_info: dict, detail_level: str = 'full') -> str:
    """Create detailed description for a single file."""
    basename = file_info['basename']
    intent = file_info['change_intent']
    adds = file_info['additions']
    dels = file_info['deletions']
    
    if detail_level == 'minimal':
        return f"• {basename}: {intent} (+{adds}/-{dels})"
    
    parts = [f"\n📄 {basename}"]
    parts.append(f"   Type: {intent}")
    parts.append(f"   Stats: +{adds}/-{dels} lines")
    
    if detail_level in ['full', 'medium']:
        added = file_info.get('added_snippets', [])
        removed = file_info.get('removed_snippets', [])
        
        if added:
            parts.append("   Changes:")
            snippet_count = 8 if detail_level == 'full' else 4
            for snippet in added[:snippet_count]:
                parts.append(f"     • {snippet}")
        
        if removed and detail_level == 'full':
            for snippet in removed[:2]:
                parts.append(f"     • {snippet}")
    
    return '\n'.join(parts)


def _create_chunked_context_for_documentation(diff_summary: dict) -> list:
    """Create multiple chunks of context for multi-pass documentation generation."""
    prioritized = _prioritize_files_for_documentation(diff_summary['files'])
    chunks = []
    
    # Chunk 1: Overview + Critical Files (detailed)
    chunk1_parts = []
    chunk1_parts.append(f"📊 DOCUMENTATION OVERVIEW")
    chunk1_parts.append(f"Total Files: {diff_summary['total_files']}")
    chunk1_parts.append(f"Changes: +{diff_summary['total_additions']} -{diff_summary['total_deletions']} lines\n")
    
    if prioritized['critical']:
        chunk1_parts.append(f"🔥 CRITICAL FILES ({len(prioritized['critical'])} files with >100 lines changed):")
        for file_info in prioritized['critical']:
            chunk1_parts.append(_create_file_detail(file_info, 'full'))
    
    if prioritized['important']:
        chunk1_parts.append(f"\n📌 IMPORTANT FILES ({len(prioritized['important'])} files with 20-100 lines):")
        for file_info in prioritized['important'][:8]:
            chunk1_parts.append(_create_file_detail(file_info, 'medium'))
    
    chunk1 = '\n'.join(chunk1_parts)
    if len(chunk1) > MAX_TOTAL_CONTEXT:
        chunk1 = chunk1[:MAX_TOTAL_CONTEXT] + "\n[Truncated]"
    chunks.append(chunk1)
    
    # Chunk 2: Moderate + Minor Files (if needed)
    if prioritized['moderate'] or prioritized['minor']:
        chunk2_parts = []
        chunk2_parts.append(f"📊 ADDITIONAL FILES")
        
        if prioritized['moderate']:
            chunk2_parts.append(f"\n📝 MODERATE FILES ({len(prioritized['moderate'])} files with 5-20 lines):")
            for file_info in prioritized['moderate'][:15]:
                chunk2_parts.append(_create_file_detail(file_info, 'minimal'))
        
        if prioritized['minor']:
            chunk2_parts.append(f"\n✨ MINOR FILES ({len(prioritized['minor'])} files with <5 lines):")
            minor_names = [f['basename'] for f in prioritized['minor'][:20]]
            chunk2_parts.append(f"Files: {', '.join(minor_names)}")
        
        chunk2 = '\n'.join(chunk2_parts)
        if len(chunk2) > MAX_TOTAL_CONTEXT:
            chunk2 = chunk2[:MAX_TOTAL_CONTEXT] + "\n[Truncated]"
        chunks.append(chunk2)
    
    return chunks


def _validate_and_fill_sections(sections: dict, diff_summary: dict) -> dict:
    """Validate sections and fill in missing ones with intelligent fallback."""
    prioritized = _prioritize_files_for_documentation(diff_summary['files'])
    
    # Check and fill SUMMARY
    if not sections.get('summary') or len(sections['summary']) < 50:
        sections['summary'] = f"This update encompasses {diff_summary['total_files']} files with significant changes totaling {diff_summary['total_additions']} additions and {diff_summary['total_deletions']} deletions. The changes primarily focus on "
        if prioritized['critical']:
            sections['summary'] += f"critical modifications to {len(prioritized['critical'])} major files, "
        sections['summary'] += "enhancing system functionality, improving code structure, and implementing new features that advance the overall project capabilities."
    
    # Check and fill CHANGES - Include ALL priority levels
    if not sections.get('changes') or len(sections['changes']) < 100:
        changes_parts = []
        
        # Critical files - detailed explanation
        if prioritized['critical']:
            changes_parts.append(f"Critical files with major modifications include {', '.join([f['basename'] for f in prioritized['critical'][:3]])}. ")
            for file_info in prioritized['critical'][:3]:
                changes_parts.append(f"The {file_info['basename']} file underwent {file_info['change_intent']} with {file_info['additions']} lines added and {file_info['deletions']} lines removed. ")
        
        # Important files - medium detail
        if prioritized['important']:
            changes_parts.append(f"Important modifications were made to {len(prioritized['important'])} files. ")
            for file_info in prioritized['important'][:3]:
                changes_parts.append(f"The {file_info['basename']} experienced {file_info['change_intent']} affecting {file_info['total_changes']} lines. ")
        
        # Moderate files - brief explanation
        if prioritized['moderate']:
            changes_parts.append(f"Moderate changes affected {len(prioritized['moderate'])} files including {', '.join([f['basename'] for f in prioritized['moderate'][:4]])}. ")
            changes_parts.append(f"These files received targeted updates with an average of {sum(f['total_changes'] for f in prioritized['moderate']) // len(prioritized['moderate'])} lines changed per file, focusing on refinements and adjustments. ")
        
        # Minor files - summary
        if prioritized['minor']:
            changes_parts.append(f"Additionally, {len(prioritized['minor'])} files received minor updates with small-scale adjustments. ")
            if len(prioritized['minor']) <= 5:
                changes_parts.append(f"These include {', '.join([f['basename'] for f in prioritized['minor']])} with minimal but necessary changes. ")
            else:
                changes_parts.append(f"Key files include {', '.join([f['basename'] for f in prioritized['minor'][:5]])} among others, each with focused modifications. ")
        
        sections['changes'] = ' '.join(changes_parts) if changes_parts else "Multiple files were updated with enhancements to system functionality and code improvements."
    
    # Check and fill TECHNICAL - Extract from ALL files
    if not sections.get('technical') or len(sections['technical']) < 100:
        tech_parts = []
        
        # Extract function/class names from ALL priority levels
        functions_found = []
        imports_found = []
        all_files = prioritized['critical'] + prioritized['important'] + prioritized['moderate']
        
        for file_info in all_files:
            for snippet in file_info.get('added_snippets', [])[:3]:
                if 'function' in snippet.lower() or 'class' in snippet.lower():
                    functions_found.append(snippet)
                elif 'import' in snippet.lower():
                    imports_found.append(snippet)
        
        # Technical implementation details
        if functions_found:
            tech_parts.append(f"The implementation includes technical additions such as {', '.join(functions_found[:5])}. ")
        
        if imports_found:
            tech_parts.append(f"New dependencies were integrated including {', '.join(imports_found[:3])}. ")
        
        # General technical description based on file types
        file_types = {}
        for f in diff_summary['files']:
            ftype = f['file_type']
            file_types[ftype] = file_types.get(ftype, 0) + 1
        
        if file_types:
            tech_parts.append(f"The technical scope spans {', '.join([f'{count} {ftype}' for ftype, count in file_types.items()])} files. ")
        
        tech_parts.append("The changes involve refactoring existing code structures, implementing new functional components, and optimizing algorithms for better performance. ")
        tech_parts.append("Code architecture improvements include better separation of concerns, enhanced modularity, and improved error handling patterns. ")
        tech_parts.append("The technical approach emphasizes maintainable code patterns, comprehensive testing support, and efficient resource utilization across all modified components.")
        sections['technical'] = ' '.join(tech_parts)
    
    # Check and fill IMPACT
    if not sections.get('impact') or len(sections['impact']) < 50:
        sections['impact'] = "These changes deliver significant business value by improving system reliability, enhancing user experience, and enabling new capabilities. The modifications reduce technical debt, increase code maintainability, and provide a solid foundation for future development. Overall system performance and scalability are positively impacted through these strategic improvements."
    
    # Check and fill RECOMMENDATIONS
    if not sections.get('recommendations') or len(sections['recommendations']) < 50:
        sections['recommendations'] = "Moving forward, it is recommended to conduct thorough testing of the modified components to ensure stability. Monitor system performance metrics to validate the improvements. Consider documenting the new features and changes for team reference. Plan for code review sessions to ensure quality standards are maintained. Implement continuous integration checks to catch potential issues early."
    
    return sections


def _generate_documentation_with_llm(context_chunks: list, chunk_index: int, total_chunks: int, diff_summary: dict) -> dict:
    """Generate documentation section using LLM with chunked context."""
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.4,
            max_tokens=4000
        )
        
        context = context_chunks[chunk_index]
        
        if chunk_index == 0:
            # First chunk: Executive Summary, Detailed Changes, Technical Implementation
            prompt = f"""You are a technical documentation expert. Analyze these git changes and write comprehensive documentation.

{context}

This is chunk {chunk_index + 1} of {total_chunks} (focusing on CRITICAL and IMPORTANT files).

You MUST write ALL THREE sections below. Each section is MANDATORY - do NOT skip any:

========================================
EXECUTIVE SUMMARY
========================================
Write 3-4 detailed paragraphs explaining:
- What changes were made overall
- Why these changes were necessary
- The business/technical significance
- Overall impact on the system

========================================
DETAILED CHANGES
========================================
For EACH major file mentioned above, write detailed paragraphs explaining:
- What specific modifications were made
- What was added or removed
- The purpose and context of each change
- How files relate to each other
Use actual file names and be specific.

========================================
TECHNICAL IMPLEMENTATION
========================================
Explain the technical details in depth:
- Specific functions, classes, or methods that were added/modified (use actual names from the context)
- Logic changes and algorithmic improvements
- New dependencies, imports, or integrations
- Technical patterns and architectures used
- Code structure and organization changes
Write 3-4 technical paragraphs with specific details.

CRITICAL REQUIREMENTS - FOLLOW EXACTLY:
1. You MUST include ALL THREE sections - NEVER skip TECHNICAL IMPLEMENTATION
2. Write section headers EXACTLY as shown with equal signs
3. Write 3-4 paragraphs minimum per section
4. Use complete sentences and flowing paragraphs - NO bullet points
5. NO markdown symbols (#, *, -, ##, [])
6. NO numbering (1., 2., 3.)
7. Be specific - use actual names from the context above
8. Each section must be substantial (at least 200 characters)"""
        else:
            # Subsequent chunks: Additional details
            prompt = f"""Continue analyzing these additional git changes for comprehensive documentation.

{context}

This is chunk {chunk_index + 1} of {total_chunks} (MODERATE and MINOR files).

Write additional documentation section:

========================================
ADDITIONAL CHANGES
========================================
Explain the moderate and minor file changes in detail:
- Group similar files together
- Describe what was updated in each group
- Explain the purpose of these additional changes
- How they complement the major changes
Write 2-3 detailed paragraphs.

CRITICAL REQUIREMENTS:
1. Write the section header EXACTLY as shown with equal signs
2. Write in complete paragraphs - NO bullet points
3. NO markdown symbols (#, *, -, ##, [])
4. Be concise but informative
5. Group similar changes together"""
        
        response = llm.invoke(prompt)
        
        if not response.content or len(response.content.strip()) < 100:
            return None
        
        return parse_ai_analysis(response.content.strip())
        
    except Exception as e:
        print(f"⚠ LLM error for chunk {chunk_index + 1}: {str(e)}")
        return None


# Helper functions for documentation generation
def parse_ai_analysis(text: str) -> dict:
    """Parse AI-generated analysis text into structured sections."""
    sections = {
        "summary": "",
        "changes": "",
        "technical": "",
        "impact": "",
        "recommendations": ""
    }
    
    # If text is too short or empty, return with default message
    if not text or len(text.strip()) < 50:
        sections["summary"] = "Analysis could not be generated. Please review the changes manually."
        return sections
    
    # Split text into lines for processing
    lines = text.split('\n')
    current_section = "summary"
    current_content = []
    
    for line in lines:
        line_stripped = line.strip()
        line_upper = line_stripped.upper()
        
        # Skip empty lines at section boundaries
        if not line_stripped:
            if current_content:
                current_content.append("")  # Preserve paragraph breaks
            continue
        
        # Detect section headers (more flexible matching)
        if any(keyword in line_upper for keyword in ["EXECUTIVE SUMMARY", "SUMMARY", "OVERVIEW"]):
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = "summary"
            current_content = []
            continue
        elif any(keyword in line_upper for keyword in ["DETAILED CHANGES", "CHANGES MADE", "WHAT CHANGED"]):
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = "changes"
            current_content = []
            continue
        elif any(keyword in line_upper for keyword in ["TECHNICAL IMPLEMENTATION", "IMPLEMENTATION", "TECHNICAL DETAILS"]):
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = "technical"
            current_content = []
            continue
        elif any(keyword in line_upper for keyword in ["BUSINESS IMPACT", "IMPACT", "VALUE"]):
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = "impact"
            current_content = []
            continue
        elif any(keyword in line_upper for keyword in ["RECOMMENDATIONS", "FUTURE", "NEXT STEPS", "SUGGESTIONS"]):
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = "recommendations"
            current_content = []
            continue
        
        # Add content line (skip obvious section markers like "1.", "2.", etc.)
        if not line_stripped.startswith(('1.', '2.', '3.', '4.', '5.', '---', '===', '###')):
            current_content.append(line)
    
    # Add last section
    if current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    # If all sections are empty, put everything in summary
    if all(not v for v in sections.values()):
        sections["summary"] = text
        sections["changes"] = "Please refer to the summary section for details."
        sections["technical"] = "Technical details are included in the overall analysis."
    
    return sections


def create_fallback_documentation(diff_summary: dict) -> dict:
    """Create basic documentation when LLM fails."""
    files = diff_summary['files']
    total_files = diff_summary['total_files']
    
    summary = f"This update includes changes to {total_files} file(s) with {diff_summary['total_additions']} additions and {diff_summary['total_deletions']} deletions. "
    
    # Analyze file types
    file_types = {}
    for f in files:
        ftype = f['file_type']
        file_types[ftype] = file_types.get(ftype, 0) + 1
    
    if file_types:
        summary += "Affected areas: " + ", ".join([f"{count} {ftype} file(s)" for ftype, count in file_types.items()]) + ". "
    
    summary += "These changes improve the codebase functionality and structure."
    
    # Build changes description with prioritization
    prioritized = _prioritize_files_for_documentation(files)
    changes_parts = []
    
    if prioritized['critical']:
        changes_parts.append(f"Critical Changes ({len(prioritized['critical'])} files):")
        for file_info in prioritized['critical']:
            changes_parts.append(f"\n{file_info['file']}: {file_info['change_intent']} with {file_info['total_changes']} lines modified.")
    
    if prioritized['important']:
        changes_parts.append(f"\n\nImportant Changes ({len(prioritized['important'])} files):")
        for file_info in prioritized['important'][:5]:
            changes_parts.append(f"\n{file_info['file']}: {file_info['change_intent']}.")
    
    if prioritized['moderate'] or prioritized['minor']:
        total_other = len(prioritized['moderate']) + len(prioritized['minor'])
        changes_parts.append(f"\n\nAdditional {total_other} files with minor to moderate changes.")
    
    return {
        "summary": summary,
        "changes": '\n'.join(changes_parts) if changes_parts else "Multiple files were updated to improve system functionality.",
        "technical": "The implementation includes code improvements, new features, and optimizations that enhance system performance and maintainability.",
        "impact": "These changes positively impact the system by improving code quality, adding new capabilities, and ensuring better long-term maintainability.",
        "recommendations": "Continue monitoring the system after deployment. Consider adding tests for new functionality. Review performance metrics to ensure optimizations are effective."
    }


@tool
def generate_version_documentation() -> dict:
    """Generate detailed PDF documentation of current changes based on git diff analysis.
    Automatically detects unstaged and staged changes (just like git commit does).
    Creates a professional PDF with AI-powered analysis.
    
    Returns:
        Dictionary with status, message, and path to generated PDF.
    """
    output_file = "version_control_doc.pdf"
    
    try:
        # First check if it's a git repository
        check_repo = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if check_repo.returncode != 0:
            return {
                "status": "error",
                "message": "❌ Not a git repository\n💡 Solution: Run 'git init' first"
            }
        
        # Get git diff output for CURRENT changes (staged + unstaged)
        # First try staged changes
        diff_result = subprocess.run(
            ['git', 'diff', '--cached'], 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        git_diff_output = diff_result.stdout.strip()
        diff_type = "staged changes"
        
        # If no staged changes, check unstaged changes
        if not git_diff_output:
            diff_result = subprocess.run(
                ['git', 'diff'], 
                capture_output=True, 
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            git_diff_output = diff_result.stdout.strip()
            diff_type = "unstaged changes"
        
        # If still no changes, check if there are untracked files
        if not git_diff_output:
            status_result = subprocess.run(
                ['git', 'status', '--short'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            if status_result.stdout.strip():
                return {
                    "status": "error",
                    "message": "❌ Only untracked files found. Please add files first.\n💡 Solution: Run 'git add .' to stage files, then generate documentation"
                }
            else:
                return {
                    "status": "error",
                    "message": "❌ No changes detected\n💡 Solution: Make changes to files first, then generate documentation"
                }
        
        # Get list of changed files
        files_result = subprocess.run(
            ['git', 'diff', '--name-status', '--cached'] if diff_type == "staged changes" else ['git', 'diff', '--name-status'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        changed_files = []
        if files_result.stdout.strip():
            for line in files_result.stdout.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    status = parts[0]
                    filename = parts[1]
                    change_type = {
                        'M': 'Modified',
                        'A': 'Added',
                        'D': 'Deleted',
                        'R': 'Renamed',
                        'C': 'Copied'
                    }.get(status[0], 'Modified')
                    changed_files.append({
                        'file': filename,
                        'type': change_type
                    })
        
        # Get repository name
        repo_result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        repo_name = os.path.basename(repo_result.stdout.strip()) if repo_result.returncode == 0 else "Repository"
        
        # Get current branch
        branch_result = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        current_branch = branch_result.stdout.strip() or "main"
        
        # Parse diff with intelligent extraction (like commit tool)
        print("🔍 Parsing git diff with intelligent extraction...")
        diff_summary = _parse_git_diff_for_documentation(git_diff_output)
        
        print(f"📊 Detected changes in {diff_summary['total_files']} file(s):")
        print(f"   Total: +{diff_summary['total_additions']} -{diff_summary['total_deletions']} lines\n")
        
        # Prioritize files
        prioritized = _prioritize_files_for_documentation(diff_summary['files'])
        print(f"🎯 Priority breakdown:")
        print(f"   🔥 Critical (>100 lines): {len(prioritized['critical'])} files")
        print(f"   📌 Important (20-100): {len(prioritized['important'])} files")
        print(f"   📝 Moderate (5-20): {len(prioritized['moderate'])} files")
        print(f"   ✨ Minor (<5): {len(prioritized['minor'])} files\n")
        
        # Create chunked context for multi-pass generation
        print("📦 Creating intelligent chunks for LLM...")
        context_chunks = _create_chunked_context_for_documentation(diff_summary)
        print(f"✅ Created {len(context_chunks)} chunk(s) for processing\n")
        
        # Use LLM to analyze changes (multi-pass if needed)
        print("🤖 Generating documentation with AI...")
        analysis_data = {"summary": "", "changes": "", "technical": "", "impact": "", "recommendations": ""}
        
        try:
            for idx, chunk in enumerate(context_chunks):
                print(f"   Processing chunk {idx + 1}/{len(context_chunks)} ({len(chunk):,} chars)...")
                
                chunk_result = _generate_documentation_with_llm(context_chunks, idx, len(context_chunks), diff_summary)
                
                if chunk_result:
                    # Merge results
                    if idx == 0:
                        # First chunk has main sections
                        analysis_data["summary"] = chunk_result.get("summary", "")
                        analysis_data["changes"] = chunk_result.get("changes", "")
                        analysis_data["technical"] = chunk_result.get("technical", "")
                        analysis_data["impact"] = chunk_result.get("impact", "")
                        analysis_data["recommendations"] = chunk_result.get("recommendations", "")
                    else:
                        # Subsequent chunks append to changes
                        additional = chunk_result.get("changes", "") or chunk_result.get("summary", "")
                        if additional:
                            analysis_data["changes"] += "\n\n" + additional
            
            # Validate and fill missing sections
            print("🔍 Validating documentation sections...")
            analysis_data = _validate_and_fill_sections(analysis_data, diff_summary)
            
            # Final check
            missing_sections = [k for k, v in analysis_data.items() if not v or len(v) < 30]
            if missing_sections:
                print(f"⚠ Filled missing sections: {', '.join(missing_sections)}")
            
            print("✅ Documentation generated and validated successfully\n")
            
        except Exception as e:
            print(f"⚠️  LLM Error: {str(e)}")
            analysis_data = create_fallback_documentation(diff_summary)
        
        # Generate PDF
        print("📄 Generating PDF documentation...")
        pdf_path = os.path.abspath(output_file)
        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                               topMargin=0.75*inch, bottomMargin=0.75*inch,
                               leftMargin=0.75*inch, rightMargin=0.75*inch)
        
        # Container for PDF elements
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#0d47a1'),
            spaceAfter=10,
            spaceBefore=16,
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderColor=colors.HexColor('#0d47a1'),
            borderPadding=5,
            backColor=colors.HexColor('#e3f2fd')
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            leading=16,
            spaceAfter=10,
            alignment=TA_LEFT,
            fontName='Helvetica',
            textColor=colors.HexColor('#212121')
        )
        
        # Title Page
        story.append(Spacer(1, 1.5*inch))
        story.append(Paragraph("📚 Version Control", title_style))
        story.append(Paragraph("Documentation Report", title_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Info table
        info_data = [
            ['Repository:', repo_name],
            ['Branch:', current_branch],
            ['Analysis Type:', diff_type.title()],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Files Changed:', str(diff_summary['total_files'])],
            ['Lines Added:', f"+{diff_summary['total_additions']}"],
            ['Lines Deleted:', f"-{diff_summary['total_deletions']}"]
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#0d47a1')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#90caf9')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(info_table)
        story.append(PageBreak())
        
        # Table of Contents
        story.append(Paragraph("📋 Table of Contents", heading_style))
        story.append(Spacer(1, 0.2*inch))
        toc_items = [
            "1. Executive Summary",
            "2. Files Changed",
            "3. Detailed Changes Analysis",
            "4. Technical Implementation",
            "5. Business Impact",
            "6. Future Recommendations"
        ]
        for item in toc_items:
            story.append(Paragraph(f"   {item}", body_style))
        story.append(PageBreak())
        
        # 1. Executive Summary
        story.append(Paragraph("1. 📊 Executive Summary", heading_style))
        story.append(Spacer(1, 0.15*inch))
        if analysis_data.get("summary"):
            for para in analysis_data["summary"].split('\n\n'):
                if para.strip():
                    safe_para = para.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(safe_para, body_style))
                    story.append(Spacer(1, 0.12*inch))
        else:
            story.append(Paragraph("This document provides a comprehensive overview of the recent changes made to the codebase.", body_style))
        story.append(Spacer(1, 0.4*inch))
        
        # 2. Changed Files Summary
        story.append(Paragraph("2. 📁 Files Changed", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Prepare files for table (using diff_summary structure)
        if diff_summary['files']:
            files_data = [['#', 'Priority', 'File Path', 'Changes']]
            
            # Add critical files
            for idx, file_info in enumerate(prioritized['critical'], 1):
                priority_icon = '🔥'
                changes = f"+{file_info['additions']}/-{file_info['deletions']}"
                files_data.append([str(idx), f"{priority_icon} Critical", file_info['file'], changes])
            
            # Add important files
            offset = len(prioritized['critical'])
            for idx, file_info in enumerate(prioritized['important'], offset + 1):
                priority_icon = '📌'
                changes = f"+{file_info['additions']}/-{file_info['deletions']}"
                files_data.append([str(idx), f"{priority_icon} Important", file_info['file'], changes])
            
            # Add some moderate files
            offset = len(prioritized['critical']) + len(prioritized['important'])
            for idx, file_info in enumerate(prioritized['moderate'][:10], offset + 1):
                priority_icon = '📝'
                changes = f"+{file_info['additions']}/-{file_info['deletions']}"
                files_data.append([str(idx), f"{priority_icon} Moderate", file_info['file'], changes])
            
            files_table = Table(files_data, colWidths=[0.4*inch, 1.1*inch, 3.3*inch, 1.2*inch])
            files_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(files_table)
            
            # Add summary of remaining files
            total_shown = len(prioritized['critical']) + len(prioritized['important']) + min(10, len(prioritized['moderate']))
            remaining = diff_summary['total_files'] - total_shown
            
            if remaining > 0:
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph(f"<i>... and {remaining} more files (minor changes)</i>", body_style))
        
        story.append(Spacer(1, 0.4*inch))
        
        # 3. Detailed Changes
        story.append(Paragraph("3. 🔍 Detailed Changes Analysis", heading_style))
        story.append(Spacer(1, 0.15*inch))
        if analysis_data.get("changes"):
            for para in analysis_data["changes"].split('\n\n'):
                if para.strip():
                    safe_para = para.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(safe_para, body_style))
                    story.append(Spacer(1, 0.12*inch))
        story.append(Spacer(1, 0.4*inch))
        
        # 4. Technical Implementation
        story.append(Paragraph("4. ⚙️ Technical Implementation", heading_style))
        story.append(Spacer(1, 0.15*inch))
        if analysis_data.get("technical"):
            for para in analysis_data["technical"].split('\n\n'):
                if para.strip():
                    safe_para = para.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(safe_para, body_style))
                    story.append(Spacer(1, 0.12*inch))
        story.append(Spacer(1, 0.4*inch))
        
        # 5. Business Impact
        story.append(Paragraph("5. 💼 Business Impact", heading_style))
        story.append(Spacer(1, 0.15*inch))
        if analysis_data.get("impact"):
            for para in analysis_data["impact"].split('\n\n'):
                if para.strip():
                    safe_para = para.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(safe_para, body_style))
                    story.append(Spacer(1, 0.12*inch))
        else:
            story.append(Paragraph("The changes contribute to improving the overall system functionality and maintainability.", body_style))
        story.append(Spacer(1, 0.4*inch))
        
        # 6. Recommendations
        story.append(Paragraph("6. 💡 Future Recommendations", heading_style))
        story.append(Spacer(1, 0.15*inch))
        if analysis_data.get("recommendations"):
            for para in analysis_data["recommendations"].split('\n\n'):
                if para.strip():
                    safe_para = para.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(safe_para, body_style))
                    story.append(Spacer(1, 0.12*inch))
        else:
            story.append(Paragraph("Continue monitoring the changes and ensure proper testing before deployment.", body_style))
        
        # Build PDF
        doc.build(story)
        
        return {
            "status": "success",
            "message": f"✅ Documentation generated successfully!\n📄 File: {pdf_path}",
            "pdf_path": pdf_path
        }
        
    except ImportError:
        return {
            "status": "error",
            "message": "❌ reportlab library not installed\n💡 Solution: Install it with: pip install reportlab"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"❌ Failed to generate documentation: {str(e)}"
        }