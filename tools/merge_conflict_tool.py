"""Advanced Merge Conflict Resolution Tool - Refactored for Better UX"""
import subprocess
import os
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

load_dotenv()


class ResolutionStrategy(Enum):
    """Available resolution strategies"""
    INTERACTIVE = "interactive"
    CURRENT = "current"
    INCOMING = "incoming"
    BOTH = "both"


@dataclass
class ConflictRegion:
    """Represents a single conflict region in a file"""
    file_path: str
    start_line: int
    end_line: int
    current_branch: str
    incoming_branch: str
    current_content: List[str]
    incoming_content: List[str]
    base_content: List[str]
    
    @property
    def current_text(self) -> str:
        return '\n'.join(self.current_content)
    
    @property
    def incoming_text(self) -> str:
        return '\n'.join(self.incoming_content)
    
    @property
    def base_text(self) -> str:
        return '\n'.join(self.base_content)


class GitOperations:
    """Handles all Git-related operations"""
    
    @staticmethod
    def get_conflicted_files() -> List[str]:
        """Get list of files with merge conflicts"""
        result = subprocess.run(
            ['git', 'diff', '--name-only', '--diff-filter=U'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')
        return []
    
    @staticmethod
    def has_conflicts() -> bool:
        """Check if repository has any merge conflicts"""
        status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        conflict_markers = ['UU', 'AA', 'DD', 'AU', 'UA', 'DU', 'UD']
        return any(
            line.startswith(tuple(conflict_markers)) 
            for line in status_result.stdout.strip().split('\n') 
            if line
        )
    
    @staticmethod
    def stage_file(file_path: str) -> bool:
        """Stage a resolved file"""
        result = subprocess.run(
            ['git', 'add', file_path],
            capture_output=True
        )
        return result.returncode == 0
    
    @staticmethod
    def get_branch_names() -> Tuple[str, str]:
        """Get current and incoming branch names"""
        # Get current branch
        current = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True
        ).stdout.strip()
        
        # Try to get merge head
        incoming = "MERGE_HEAD"
        try:
            with open('.git/MERGE_HEAD', 'r') as f:
                incoming = f.read().strip()[:8]
        except:
            pass
        
        return current, incoming


class ConflictParser:
    """Parses conflict markers from file content"""
    
    @staticmethod
    def parse_file(file_path: str) -> List[ConflictRegion]:
        """Parse all conflict regions in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}")
        
        return ConflictParser._parse_content(content, file_path)
    
    @staticmethod
    def _parse_content(content: str, file_path: str) -> List[ConflictRegion]:
        """Parse conflict markers from content"""
        conflicts = []
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            if lines[i].startswith('<<<<<<<'):
                conflict = ConflictParser._parse_single_conflict(lines, i, file_path)
                if conflict:
                    conflicts.append(conflict)
                    i = conflict.end_line
            i += 1
        
        return conflicts
    
    @staticmethod
    def _parse_single_conflict(lines: List[str], start: int, file_path: str) -> Optional[ConflictRegion]:
        """Parse a single conflict region"""
        current_branch = lines[start].replace('<<<<<<< ', '').strip()
        current_content = []
        incoming_content = []
        base_content = []
        
        i = start + 1
        
        # Parse current branch content
        while i < len(lines) and not lines[i].startswith('=======') and not lines[i].startswith('|||||||'):
            current_content.append(lines[i])
            i += 1
        
        # Check for diff3 style (with base)
        if i < len(lines) and lines[i].startswith('|||||||'):
            i += 1
            while i < len(lines) and not lines[i].startswith('======='):
                base_content.append(lines[i])
                i += 1
        
        # Skip separator
        if i < len(lines) and lines[i].startswith('======='):
            i += 1
        
        # Parse incoming content
        while i < len(lines) and not lines[i].startswith('>>>>>>>'):
            incoming_content.append(lines[i])
            i += 1
        
        # Get incoming branch name
        incoming_branch = "MERGE_HEAD"
        if i < len(lines) and lines[i].startswith('>>>>>>>'):
            incoming_branch = lines[i].replace('>>>>>>> ', '').strip()
        
        return ConflictRegion(
            file_path=file_path,
            start_line=start,
            end_line=i,
            current_branch=current_branch,
            incoming_branch=incoming_branch,
            current_content=current_content,
            incoming_content=incoming_content,
            base_content=base_content
        )


class AIAnalyzer:
    """Handles AI-powered conflict analysis"""
    
    def __init__(self):
        model_id = os.getenv("HF_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500,
            temperature=0.1,
            top_k=50,
            do_sample=True
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def analyze_conflict(self, conflict: ConflictRegion) -> str:
        """
        Analyze a conflict and explain the differences.
        CRITICAL: Only returns explanation, NEVER generates code.
        """
        file_ext = os.path.splitext(conflict.file_path)[1]
        
        prompt = f"""You are analyzing a merge conflict. Your job is to EXPLAIN the differences, NOT to write code.

FILE: {conflict.file_path} ({file_ext})

CURRENT BRANCH ({conflict.current_branch}):
{conflict.current_text}

INCOMING BRANCH ({conflict.incoming_branch}):
{conflict.incoming_text}

Provide a 2-3 sentence analysis covering:
1. What does the CURRENT version do?
2. What does the INCOMING version do?
3. What is the key difference?

DO NOT generate any code. DO NOT suggest a resolution. ONLY explain what changed.
Keep it concise and clear."""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip() if response.content else "Unable to analyze."
        except Exception as e:
            return f"Analysis failed: {str(e)}"


class IntelligentMerger:
    """Uses LLM to intelligently merge both versions"""
    
    def __init__(self):
        model_id = os.getenv("HF_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1000,
            temperature=0.1,
            top_k=50,
            do_sample=True
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def merge_both(self, conflict: ConflictRegion, analysis: str, max_retries: int = 2) -> Tuple[str, bool]:
        """
        Use LLM to intelligently merge both versions into valid code.
        Includes self-reflection to fix syntax errors.
        
        Args:
            conflict: The conflict region
            analysis: The AI analysis explaining the differences
            max_retries: Maximum attempts to fix syntax errors
            
        Returns:
            Tuple of (merged_code, is_valid)
        """
        file_ext = os.path.splitext(conflict.file_path)[1]
        
        prompt = f"""You are merging two versions of code into one valid implementation.

            FILE: {conflict.file_path} ({file_ext})
            CONTEXT: {analysis}

            CURRENT VERSION:
            {conflict.current_text}

            INCOMING VERSION:
            {conflict.incoming_text}

            TASK: Create ONE valid merged version that intelligently combines both.

            RULES:
            1. Output ONLY the merged code, NO explanations or markdown.
            2. Ensure syntax is 100% valid.
            3. If both versions do the same thing differently, choose the better one.
            4. If they provide different functionality, preserve both if it makes sense.
            5. If merging both creates invalid code (e.g., duplicate returns), choose the better version.
            6. Remove any conflict markers (<<<<<<<, =======, >>>>>>>).
            7. Maintain proper indentation and formatting.
            8. Do NOT introduce new functions, classes, or variables that are not in either version.
            9. If uncertain, prefer the incoming version exactly as written.
            10. Do NOT include backticks or markdown formatting.
            11. Preserve indentation correctly for Python/YAML code.
            12. Use original Function name don't create new name like merged_code, or resolved_code and etc.

            Merged code:"""

        try:
            response = self.llm.invoke(prompt)
            merged = response.content.strip() if response.content else None
            
            if not merged:
                return conflict.incoming_text, False
            
            # Clean up any remaining markdown or markers
            merged = self._clean_output(merged)
            
            # Validate syntax
            is_valid, error = SyntaxValidator.validate(conflict.file_path, merged)
            
            # If invalid, try to fix it with reflection
            retry_count = 0
            while not is_valid and retry_count < max_retries:
                retry_count += 1
                print(f"   ⚠️  Syntax error detected, attempting fix (attempt {retry_count}/{max_retries})...")
                print(f"   Error: {error}")
                
                merged = self._fix_syntax_error(conflict, merged, error, file_ext)
                is_valid, error = SyntaxValidator.validate(conflict.file_path, merged)
            
            if is_valid:
                return merged.strip(), True
            else:
                print(f"   ❌ Could not fix syntax after {max_retries} attempts")
                return merged.strip(), False
            
        except Exception as e:
            print(f"⚠️  LLM merge failed: {e}")
            return conflict.incoming_text, False
    
    def _clean_output(self, code: str) -> str:
        """Clean LLM output of markdown and conflict markers"""
        code = re.sub(r'^```[\w]*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```$', '', code)
        code = re.sub(r'^<{7}.*?\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^={7}\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^>{7}.*?\n', '', code, flags=re.MULTILINE)
        return code.strip()
    
    def _fix_syntax_error(self, conflict: ConflictRegion, broken_code: str, error: str, file_ext: str) -> str:
        """Use LLM to fix syntax error in merged code"""
        fix_prompt = f"""The following merged code has a syntax error. Fix it.

FILE TYPE: {file_ext}
ERROR: {error}

BROKEN CODE:
```
{broken_code}
```

ORIGINAL VERSIONS FOR REFERENCE:
CURRENT:
```
{conflict.current_text}
```

INCOMING:
```
{conflict.incoming_text}
```

Fix the syntax error and output ONLY the corrected code, NO explanations.

Fixed code:"""
        
        try:
            response = self.llm.invoke(fix_prompt)
            fixed = response.content.strip() if response.content else broken_code
            return self._clean_output(fixed)
        except:
            return broken_code


class SyntaxValidator:
    """Validates syntax of resolved code"""
    
    @staticmethod
    def validate_python(code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax"""
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def validate(file_path: str, code: str) -> Tuple[bool, Optional[str]]:
        """Validate syntax based on file type"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.py':
            return SyntaxValidator.validate_python(code)
        
        # For other file types, just check it's not empty
        if not code.strip():
            return False, "Resolved content is empty"
        
        return True, None


class ConflictResolver:
    """Handles conflict resolution strategies"""
    
    def __init__(self):
        self.intelligent_merger = IntelligentMerger()
        self.syntax_validator = SyntaxValidator()
    
    def resolve(self, conflict: ConflictRegion, strategy: ResolutionStrategy, analysis: str = "") -> Tuple[str, bool]:
        """Apply resolution strategy to a conflict
        
        Returns:
            Tuple of (resolved_code, is_valid)
        """
        if strategy == ResolutionStrategy.CURRENT:
            return conflict.current_text, True
        
        elif strategy == ResolutionStrategy.INCOMING:
            return conflict.incoming_text, True
        
        elif strategy == ResolutionStrategy.BOTH:
            # Use intelligent LLM-based merging with self-correction
            return self.intelligent_merger.merge_both(conflict, analysis)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def apply_resolution(self, file_path: str, conflicts: List[ConflictRegion], 
                        resolutions: List[str]) -> Tuple[bool, Optional[str]]:
        """Apply resolved content back to file with syntax validation
        
        Args:
            resolutions: List of resolved code strings (not tuples)
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            resolved_lines = []
            last_line = 0
            
            for conflict, resolution in zip(conflicts, resolutions):
                # Add lines before conflict
                resolved_lines.extend(lines[last_line:conflict.start_line])
                # Add resolved content
                resolved_lines.extend(resolution.split('\n'))
                # Move past conflict
                last_line = conflict.end_line + 1
            
            # Add remaining lines
            resolved_lines.extend(lines[last_line:])
            
            # Create the final content
            resolved_content = '\n'.join(resolved_lines)
            
            # Final validation before writing
            is_valid, error = self.syntax_validator.validate(file_path, resolved_content)
            if not is_valid:
                return False, f"Final validation failed: {error}"
            
            # Write back
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.write(resolved_content)
            
            return True, None
        except Exception as e:
            return False, f"Failed to apply resolution: {e}"


class ConflictFormatter:
    """Formats conflict information for display"""
    
    @staticmethod
    def format_conflict_display(conflict: ConflictRegion, conflict_num: int, 
                               total: int, analysis: str) -> str:
        """Format a single conflict for user review"""
        output = []
        output.append("=" * 80)
        output.append(f"📄 FILE: {conflict.file_path}")
        output.append(f"🔢 CONFLICT {conflict_num} of {total}")
        output.append("=" * 80)
        output.append("")
        
        # Current version
        output.append(f"📌 CURRENT ({conflict.current_branch}):")
        output.append("─" * 80)
        output.append(conflict.current_text if conflict.current_text.strip() else "(empty)")
        output.append("")
        
        # Incoming version
        output.append(f"📥 INCOMING ({conflict.incoming_branch}):")
        output.append("─" * 80)
        output.append(conflict.incoming_text if conflict.incoming_text.strip() else "(empty)")
        output.append("")
        
        # AI Analysis
        output.append("🤖 AI ANALYSIS:")
        output.append("─" * 80)
        output.append(analysis)
        output.append("")
        
        return "\n".join(output)
    
    @staticmethod
    def format_summary(total_files: int, total_conflicts: int, 
                      resolved: int, failed: int) -> str:
        """Format resolution summary"""
        output = []
        output.append("")
        output.append("=" * 80)
        output.append("📊 RESOLUTION SUMMARY")
        output.append("=" * 80)
        output.append(f"Total Files: {total_files}")
        output.append(f"Total Conflicts: {total_conflicts}")
        output.append(f"✅ Resolved: {resolved}")
        output.append(f"❌ Failed: {failed}")
        output.append("=" * 80)
        
        if failed == 0:
            output.append("")
            output.append("✅ All conflicts resolved successfully!")
            output.append("")
            output.append("💡 Next steps:")
            output.append("   1. Review changes: git diff --cached")
            output.append("   2. Commit: git commit -m 'Resolved merge conflicts'")
        
        return "\n".join(output)


@tool
def get_merge_conflicts() -> dict:
    """
    Show all merge conflicts with detailed AI analysis.
    
    This displays:
    - Each conflict side-by-side
    - AI explanation of what changed
    - Clear options for resolution
    
    Use this FIRST to understand conflicts before resolving.
    
    Returns:
        Detailed conflict information with AI analysis
    """
    try:
        if not GitOperations.has_conflicts():
            return {
                "status": "success",
                "message": "✅ No merge conflicts detected"
            }
        
        conflicted_files = GitOperations.get_conflicted_files()
        if not conflicted_files:
            return {
                "status": "success",
                "message": "✅ No merge conflicts found"
            }
        
        analyzer = AIAnalyzer()
        all_conflicts = []
        total_conflicts = 0
        
        output = []
        output.append("")
        output.append("=" * 80)
        output.append("🔍 MERGE CONFLICTS DETECTED")
        output.append("=" * 80)
        output.append(f"Files with conflicts: {len(conflicted_files)}")
        output.append("")
        
        for file_path in conflicted_files:
            try:
                conflicts = ConflictParser.parse_file(file_path)
                
                for idx, conflict in enumerate(conflicts, 1):
                    total_conflicts += 1
                    analysis = analyzer.analyze_conflict(conflict)
                    
                    display = ConflictFormatter.format_conflict_display(
                        conflict, idx, len(conflicts), analysis
                    )
                    output.append(display)
                    
                    all_conflicts.append({
                        'file': file_path,
                        'number': idx,
                        'analysis': analysis
                    })
                    
            except Exception as e:
                output.append(f"❌ Error analyzing {file_path}: {e}")
        
        output.append("")
        output.append("=" * 80)
        output.append("💡 RESOLUTION OPTIONS")
        output.append("=" * 80)
        output.append("")
        output.append("Choose a strategy for ALL conflicts:")
        output.append("")
        output.append("1️⃣  resolve_conflicts('current')")
        output.append("    → Keep all CURRENT branch changes")
        output.append("")
        output.append("2️⃣  resolve_conflicts('incoming')")
        output.append("    → Accept all INCOMING branch changes")
        output.append("")
        output.append("3️⃣  resolve_conflicts('both') 🤖 AI-POWERED")
        output.append("    → Intelligently merge both versions using AI")
        output.append("    → Validates syntax before applying")
        output.append("    → Best for combining different features")
        output.append("")
        output.append("4️⃣  Manual resolution")
        output.append("    → Edit files directly, then: git add <file>")
        output.append("")
        output.append("=" * 80)
        
        return {
            "status": "success",
            "message": "\n".join(output),
            "total_conflicts": total_conflicts,
            "conflicted_files": conflicted_files
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"❌ Failed to analyze conflicts: {str(e)}"
        }


@tool
def resolve_conflicts(strategy: str) -> dict:
    """
    Resolve all merge conflicts using the specified strategy.
    
    Args:
        strategy: One of:
            - 'current': Keep current branch changes
            - 'incoming': Accept incoming branch changes  
            - 'both': Use AI to intelligently merge both versions (with self-correction)
    
    Returns:
        Resolution result with summary
    """
    try:
        # Validate strategy
        try:
            strat = ResolutionStrategy(strategy.lower())
        except ValueError:
            return {
                "status": "error",
                "message": f"❌ Invalid strategy: {strategy}\n"
                          f"Valid options: current, incoming, both"
            }
        
        if strat == ResolutionStrategy.INTERACTIVE:
            return {
                "status": "error",
                "message": "❌ Use get_merge_conflicts() first to see conflicts,\n"
                          "then choose: current, incoming, or both"
            }
        
        conflicted_files = GitOperations.get_conflicted_files()
        if not conflicted_files:
            return {
                "status": "success",
                "message": "✅ No conflicts to resolve"
            }
        
        resolver = ConflictResolver()
        analyzer = AIAnalyzer() if strat == ResolutionStrategy.BOTH else None
        
        total_conflicts = 0
        resolved_count = 0
        failed_files = []
        
        print(f"\n🔧 Resolving conflicts with strategy: {strategy.upper()}")
        if strat == ResolutionStrategy.BOTH:
            print("🤖 Using AI to intelligently merge both versions...")
            print("⚙️  Will validate syntax and auto-fix errors if needed\n")
        else:
            print()
        
        for file_path in conflicted_files:
            try:
                conflicts = ConflictParser.parse_file(file_path)
                total_conflicts += len(conflicts)
                
                # Get analyses for 'both' strategy
                analyses = []
                if strat == ResolutionStrategy.BOTH and analyzer:
                    print(f"🔍 Analyzing {file_path}...")
                    for conflict in conflicts:
                        analysis = analyzer.analyze_conflict(conflict)
                        analyses.append(analysis)
                else:
                    analyses = [""] * len(conflicts)
                
                # Apply strategy to all conflicts
                print(f"🔧 Resolving {len(conflicts)} conflict(s) in {file_path}...")
                
                resolutions = []
                all_valid = True
                
                for idx, (conflict, analysis) in enumerate(zip(conflicts, analyses), 1):
                    if strat == ResolutionStrategy.BOTH:
                        print(f"   Conflict {idx}/{len(conflicts)}: Merging...")
                    
                    resolved_code, is_valid = resolver.resolve(conflict, strat, analysis)
                    resolutions.append(resolved_code)
                    
                    if not is_valid:
                        all_valid = False
                        print(f"   ⚠️  Warning: Conflict {idx} may have syntax issues")
                
                # Write back to file with final validation
                success, error = resolver.apply_resolution(file_path, conflicts, resolutions)
                
                if success:
                    if GitOperations.stage_file(file_path):
                        resolved_count += len(conflicts)
                        print(f"✅ {file_path}: Resolved {len(conflicts)} conflict(s)")
                        if strat == ResolutionStrategy.BOTH and all_valid:
                            print(f"   ✓ All merges validated successfully")
                    else:
                        failed_files.append((file_path, "Failed to stage"))
                        print(f"❌ {file_path}: Failed to stage")
                else:
                    failed_files.append((file_path, error))
                    print(f"❌ {file_path}: {error}")
                    
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"❌ {file_path}: {str(e)}")
        
        # Build detailed summary
        summary_parts = []
        summary_parts.append(ConflictFormatter.format_summary(
            len(conflicted_files),
            total_conflicts,
            resolved_count,
            len(failed_files)
        ))
        
        # Add failure details if any
        if failed_files:
            summary_parts.append("\n\n❌ Failed Files:")
            for file_path, error in failed_files:
                summary_parts.append(f"\n   • {file_path}")
                summary_parts.append(f"     Reason: {error}")
        
        return {
            "status": "success" if not failed_files else "partial",
            "message": "".join(summary_parts),
            "resolved": resolved_count,
            "total": total_conflicts
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"❌ Resolution failed: {str(e)}"
        }