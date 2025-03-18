"""
Code Index MCP Server

This MCP server allows LLMs to index, search, and analyze code from a project directory.
It provides tools for file discovery, content retrieval, and code analysis.
"""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Tuple, Any
import os
import pathlib
import json
import fnmatch
from mcp.server.fastmcp import FastMCP, Context, Image
from mcp import types

# Import the ProjectSettings class
from project_settings import ProjectSettings

# Create the MCP server
mcp = FastMCP("CodeIndexer", dependencies=["pathlib"])

# In-memory references (will be loaded from persistent storage)
file_index = {}
code_content_cache = {}
supported_extensions = [
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rb', '.php', '.swift', '.kt', '.rs', '.scala', '.sh',
    '.bash', '.html', '.css', '.scss', '.md', '.json', '.xml', '.yml', '.yaml'
]

@dataclass
class CodeIndexerContext:
    """Context for the Code Indexer MCP server."""
    base_path: str
    settings: ProjectSettings
    file_count: int = 0

@asynccontextmanager
async def indexer_lifespan(server: FastMCP) -> AsyncIterator[CodeIndexerContext]:
    """Manage the lifecycle of the Code Indexer MCP server."""
    # We will not set a default base_path
    # The user must explicitly set the project path before using the system
    base_path = ""  # Empty string to indicate no path is set
    
    # Initialize the settings manager with a temporary path
    # This will be properly set when the user calls set_project_path
    settings = ProjectSettings(base_path or os.getcwd())
    
    # Initialize the context
    context = CodeIndexerContext(
        base_path=base_path,
        settings=settings
    )
    
    # Try to load existing index and cache
    global file_index, code_content_cache
    loaded_index = settings.load_index()
    if loaded_index:
        file_index = loaded_index
        context.file_count = _count_files(file_index)
    
    loaded_cache = settings.load_cache()
    if loaded_cache:
        code_content_cache = loaded_cache
    
    try:
        # Yield the context to the server
        yield context
    finally:
        # Save index and cache on shutdown
        if file_index:
            settings.save_index(file_index)
        if code_content_cache:
            settings.save_cache(code_content_cache)

# Initialize the server with our lifespan manager
mcp = FastMCP("CodeIndexer", lifespan=indexer_lifespan)

# ----- RESOURCES -----

@mcp.resource("config://code-indexer")
def get_config() -> str:
    """Get the current configuration of the Code Indexer."""
    ctx = mcp.get_context()
    
    # Get the base path from context
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return json.dumps({
            "status": "not_configured",
            "message": "Project path not set. Please use set_project_path to set a project directory first.",
            "supported_extensions": supported_extensions
        }, indent=2)
    
    # Get file count
    file_count = ctx.request_context.lifespan_context.file_count
    
    # Get settings stats
    settings = ctx.request_context.lifespan_context.settings
    settings_stats = settings.get_stats()
    
    config = {
        "base_path": base_path,
        "supported_extensions": supported_extensions,
        "file_count": file_count,
        "settings_directory": settings.settings_path,
        "settings_stats": settings_stats
    }
    
    return json.dumps(config, indent=2)

@mcp.resource("files://{file_path}")
def get_file_content(file_path: str) -> str:
    """Get the content of a specific file."""
    ctx = mcp.get_context()
    
    # Get the base path from context
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return "Error: Project path not set. Please use set_project_path to set a project directory first."
    
    # Handle absolute paths (especially Windows paths starting with drive letters)
    if os.path.isabs(file_path) or (len(file_path) > 1 and file_path[1] == ':'):
        # Absolute paths are not allowed via this endpoint
        return f"Error: Absolute file paths like '{file_path}' are not allowed. Please use paths relative to the project root."
    
    # Normalize the file path
    norm_path = os.path.normpath(file_path)
    
    # Check for path traversal attempts
    if "..\\" in norm_path or "../" in norm_path or norm_path.startswith(".."): 
        return f"Error: Invalid file path: {file_path} (directory traversal not allowed)"
    
    # Construct the full path and verify it's within the project bounds
    full_path = os.path.join(base_path, norm_path)
    real_full_path = os.path.realpath(full_path)
    real_base_path = os.path.realpath(base_path)
    
    if not real_full_path.startswith(real_base_path):
        return f"Error: Access denied. File path must be within project directory."
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Cache the content for faster retrieval later
        code_content_cache[norm_path] = content
        
        return content
    except UnicodeDecodeError:
        return f"Error: File {file_path} appears to be a binary file or uses unsupported encoding."
    except Exception as e:
        return f"Error reading file: {e}"

@mcp.resource("structure://project")
def get_project_structure() -> str:
    """Get the structure of the project as a JSON tree."""
    ctx = mcp.get_context()
    
    # Get the base path from context
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return json.dumps({
            "status": "not_configured",
            "message": "Project path not set. Please use set_project_path to set a project directory first."
        }, indent=2)
    
    # Check if we need to refresh the index
    if not file_index:
        _index_project(base_path)
        # Update file count in context
        ctx.request_context.lifespan_context.file_count = _count_files(file_index)
        # Save updated index
        ctx.request_context.lifespan_context.settings.save_index(file_index)
    
    return json.dumps(file_index, indent=2)

@mcp.resource("settings://stats")
def get_settings_stats() -> str:
    """Get statistics about the settings directory and files."""
    ctx = mcp.get_context()
    
    # Get settings manager from context
    settings = ctx.request_context.lifespan_context.settings
    
    # Get settings stats
    stats = settings.get_stats()
    
    return json.dumps(stats, indent=2)

# ----- TOOLS -----

@mcp.tool()
def set_project_path(path: str, ctx: Context) -> str:
    """Set the base project path for indexing."""
    # Validate and normalize the path
    try:
        norm_path = os.path.normpath(path)
        abs_path = os.path.abspath(norm_path)
        
        if not os.path.exists(abs_path):
            return f"Error: Path does not exist: {abs_path}"
        
        if not os.path.isdir(abs_path):
            return f"Error: Path is not a directory: {abs_path}"
        
        # Clear existing in-memory index and cache
        global file_index, code_content_cache
        file_index.clear()
        code_content_cache.clear()
        
        # Update the base path in context
        ctx.request_context.lifespan_context.base_path = abs_path
        
        # Create a new settings manager for the new path
        ctx.request_context.lifespan_context.settings = ProjectSettings(abs_path)
        
        # Try to load existing index and cache
        loaded_index = ctx.request_context.lifespan_context.settings.load_index()
        if loaded_index:
            file_index = loaded_index
            file_count = _count_files(file_index)
            ctx.request_context.lifespan_context.file_count = file_count
            return f"Project path set to: {abs_path}. Loaded existing index with {file_count} files."
        
        # If no existing index, create a new one
        file_count = _index_project(abs_path)
        ctx.request_context.lifespan_context.file_count = file_count
        
        # Save the new index
        ctx.request_context.lifespan_context.settings.save_index(file_index)
        
        # Save project config
        config = {
            "base_path": abs_path,
            "supported_extensions": supported_extensions,
            "last_indexed": ctx.request_context.lifespan_context.settings.load_config().get('last_indexed', None)
        }
        ctx.request_context.lifespan_context.settings.save_config(config)
        
        return f"Project path set to: {abs_path}. Indexed {file_count} files."
    except Exception as e:
        return f"Error setting project path: {e}"

@mcp.tool()
def search_code(query: str, ctx: Context, extensions: Optional[List[str]] = None, case_sensitive: bool = False) -> Dict[str, List[Tuple[int, str]]]:
    """
    Search for code matches within the indexed files.
    Returns a dictionary mapping filenames to lists of (line_number, line_content) tuples.
    """
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return {"error": "Project path not set. Please use set_project_path to set a project directory first."}
    
    # Check if we need to index the project
    if not file_index:
        _index_project(base_path)
        ctx.request_context.lifespan_context.file_count = _count_files(file_index)
        ctx.request_context.lifespan_context.settings.save_index(file_index)
    
    results = {}
    
    # Filter by extensions if provided
    if extensions:
        valid_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    else:
        valid_extensions = supported_extensions
    
    # Process the search
    for file_path, _info in _get_all_files(file_index):
        # Check if the file has a supported extension
        if not any(file_path.endswith(ext) for ext in valid_extensions):
            continue
        
        try:
            # Get file content (from cache if available)
            if file_path in code_content_cache:
                content = code_content_cache[file_path]
            else:
                full_path = os.path.join(base_path, file_path)
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                code_content_cache[file_path] = content
            
            # Search for matches
            matches = []
            for i, line in enumerate(content.splitlines(), 1):
                if (case_sensitive and query in line) or (not case_sensitive and query.lower() in line.lower()):
                    matches.append((i, line.strip()))
            
            if matches:
                results[file_path] = matches
        except Exception as e:
            ctx.info(f"Error searching file {file_path}: {e}")
    
    # Save the updated cache
    ctx.request_context.lifespan_context.settings.save_cache(code_content_cache)
    
    return results

@mcp.tool()
def find_files(pattern: str, ctx: Context) -> List[str]:
    """
    Find files in the project that match the given pattern.
    Supports glob patterns like *.py or **/*.js.
    """
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return ["Error: Project path not set. Please use set_project_path to set a project directory first."]
    
    # Check if we need to index the project
    if not file_index:
        _index_project(base_path)
        ctx.request_context.lifespan_context.file_count = _count_files(file_index)
        ctx.request_context.lifespan_context.settings.save_index(file_index)
    
    matching_files = []
    for file_path, _info in _get_all_files(file_index):
        if fnmatch.fnmatch(file_path, pattern):
            matching_files.append(file_path)
    
    return matching_files

@mcp.tool()
def get_file_summary(file_path: str, ctx: Context) -> Dict[str, Any]:
    """
    Get a summary of a specific file, including:
    - Line count
    - Function/class definitions (for supported languages)
    - Import statements
    - Basic complexity metrics
    """
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return {"error": "Project path not set. Please use set_project_path to set a project directory first."}
    
    # Normalize the file path
    norm_path = os.path.normpath(file_path)
    if norm_path.startswith('..'):
        return {"error": f"Invalid file path: {file_path}"}
    
    full_path = os.path.join(base_path, norm_path)
    
    try:
        # Get file content
        if norm_path in code_content_cache:
            content = code_content_cache[norm_path]
        else:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            code_content_cache[norm_path] = content
            # Save the updated cache
            ctx.request_context.lifespan_context.settings.save_cache(code_content_cache)
        
        # Basic file info
        lines = content.splitlines()
        line_count = len(lines)
        
        # File extension for language-specific analysis
        _, ext = os.path.splitext(norm_path)
        
        summary = {
            "file_path": norm_path,
            "line_count": line_count,
            "size_bytes": os.path.getsize(full_path),
            "extension": ext,
        }
        
        # Language-specific analysis
        if ext == '.py':
            # Python analysis
            imports = []
            classes = []
            functions = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Check for imports
                if line.startswith('import ') or line.startswith('from '):
                    imports.append(line)
                
                # Check for class definitions
                if line.startswith('class '):
                    classes.append({
                        "line": i + 1,
                        "name": line.replace('class ', '').split('(')[0].split(':')[0].strip()
                    })
                
                # Check for function definitions
                if line.startswith('def '):
                    functions.append({
                        "line": i + 1,
                        "name": line.replace('def ', '').split('(')[0].strip()
                    })
            
            summary.update({
                "imports": imports,
                "classes": classes,
                "functions": functions,
                "import_count": len(imports),
                "class_count": len(classes),
                "function_count": len(functions),
            })
        
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            # JavaScript/TypeScript analysis
            imports = []
            classes = []
            functions = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Check for imports
                if line.startswith('import ') or line.startswith('require('):
                    imports.append(line)
                
                # Check for class definitions
                if line.startswith('class ') or 'class ' in line:
                    class_name = ""
                    if 'class ' in line:
                        parts = line.split('class ')[1]
                        class_name = parts.split(' ')[0].split('{')[0].split('extends')[0].strip()
                    classes.append({
                        "line": i + 1,
                        "name": class_name
                    })
                
                # Check for function definitions
                if 'function ' in line or '=>' in line:
                    functions.append({
                        "line": i + 1,
                        "content": line
                    })
            
            summary.update({
                "imports": imports,
                "classes": classes,
                "functions": functions,
                "import_count": len(imports),
                "class_count": len(classes),
                "function_count": len(functions),
            })
        
        return summary
    except Exception as e:
        return {"error": f"Error analyzing file: {e}"}

@mcp.tool()
def refresh_index(ctx: Context) -> str:
    """Refresh the project index."""
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return "Error: Project path not set. Please use set_project_path to set a project directory first."
    
    # Clear existing index
    global file_index
    file_index.clear()
    
    # Re-index the project
    file_count = _index_project(base_path)
    ctx.request_context.lifespan_context.file_count = file_count
    
    # Save the updated index
    ctx.request_context.lifespan_context.settings.save_index(file_index)
    
    # Update the last indexed timestamp in config
    config = ctx.request_context.lifespan_context.settings.load_config()
    ctx.request_context.lifespan_context.settings.save_config({
        **config,
        'last_indexed': ctx.request_context.lifespan_context.settings._get_timestamp()
    })
    
    return f"Project re-indexed. Found {file_count} files."

@mcp.tool()
def get_settings_info(ctx: Context) -> Dict[str, Any]:
    """Get information about the project settings."""
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return {
            "status": "not_configured",
            "message": "Project path not set. Please use set_project_path to set a project directory first."
        }
    
    settings = ctx.request_context.lifespan_context.settings
    
    # Get config
    config = settings.load_config()
    
    # Get stats
    stats = settings.get_stats()
    
    return {
        "settings_directory": settings.settings_path,
        "config": config,
        "stats": stats,
        "exists": os.path.exists(settings.settings_path)
    }

@mcp.tool()
def clear_settings(ctx: Context) -> str:
    """Clear all settings and cached data."""
    base_path = ctx.request_context.lifespan_context.base_path
    
    # Check if base_path is set
    if not base_path:
        return "Error: Project path not set. Please use set_project_path to set a project directory first."
    
    settings = ctx.request_context.lifespan_context.settings
    
    # Clear all settings files
    settings.clear()
    
    # Clear in-memory cache and index
    global file_index, code_content_cache
    file_index.clear()
    code_content_cache.clear()
    
    return f"All settings and cache cleared from {settings.settings_path}"

# ----- PROMPTS -----

@mcp.prompt()
def analyze_code(file_path: str = "", query: str = "") -> list[types.PromptMessage]:
    """Prompt for analyzing code in the project."""
    messages = [
        types.PromptMessage(role="user", content=types.TextContent(type="text", text=f"""I need you to analyze some code from my project. 
        
{f'Please analyze the file: {file_path}' if file_path else ''}
{f'I want to understand: {query}' if query else ''}

First, let me give you some context about the project structure. Then, I'll provide the code to analyze.
""")),
        types.PromptMessage(role="assistant", content=types.TextContent(type="text", text="I'll help you analyze the code. Let me first examine the project structure to get a better understanding of the codebase."))
    ]
    return messages

@mcp.prompt()
def code_search(query: str = "") -> types.TextContent:
    """Prompt for searching code in the project."""
    search_text = f"\"query\"" if not query else f"\"{query}\""
    return types.TextContent(type="text", text=f"""I need to search through my codebase for {search_text}.

Please help me find all occurrences of this query and explain what each match means in its context.
Focus on the most relevant files and provide a brief explanation of how each match is used in the code.

If there are too many results, prioritize the most important ones and summarize the patterns you see.""")

@mcp.prompt()
def set_project() -> list[types.PromptMessage]:
    """Prompt for setting the project path."""
    messages = [
        types.PromptMessage(role="user", content=types.TextContent(type="text", text="""
        I need to analyze code from a project, but I haven't set the project path yet. Please help me set up the project path and index the code.
        
        First, I need to specify which project directory to analyze.
        """)),
        types.PromptMessage(role="assistant", content=types.TextContent(type="text", text="""
        Before I can help you analyze any code, we need to set up the project path. This is a required first step.
        
        Please provide the full path to your project folder. For example:
        - Windows: "C:/Users/username/projects/my-project"
        - macOS/Linux: "/home/username/projects/my-project"
        
        Once you provide the path, I'll use the `set_project_path` tool to configure the code analyzer to work with your project.
        """))
    ]
    return messages

# ----- HELPER FUNCTIONS -----

def _index_project(base_path: str) -> int:
    """
    Create an index of the project files.
    Returns the number of files indexed.
    """
    file_count = 0
    file_index.clear()
    
    for root, dirs, files in os.walk(base_path):
        # Skip hidden directories and common build/dependency directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and 
                 d not in ['node_modules', 'venv', '__pycache__', 'build', 'dist']]
        
        # Create relative path from base_path
        rel_path = os.path.relpath(root, base_path)
        current_dir = file_index
        
        # Skip the '.' directory (base_path itself)
        if rel_path != '.':
            # Split the path and navigate/create the tree
            path_parts = rel_path.replace('\\', '/').split('/')
            for part in path_parts:
                if part not in current_dir:
                    current_dir[part] = {}
                current_dir = current_dir[part]
        
        # Add files to current directory
        for file in files:
            # Skip hidden files and files with unsupported extensions
            _, ext = os.path.splitext(file)
            if file.startswith('.') or ext not in supported_extensions:
                continue
                
            # Store file information
            file_path = os.path.join(rel_path, file).replace('\\', '/')
            if rel_path == '.':
                file_path = file
                
            current_dir[file] = {
                "type": "file",
                "path": file_path,
                "ext": ext
            }
            file_count += 1
            
    return file_count

def _count_files(directory: Dict) -> int:
    """
    Count the number of files in the index.
    """
    count = 0
    for name, value in directory.items():
        if isinstance(value, dict):
            if "type" in value and value["type"] == "file":
                count += 1
            else:
                count += _count_files(value)
    return count

def _get_all_files(directory: Dict, prefix: str = "") -> List[Tuple[str, Dict]]:
    """
    Recursively get all files from the directory structure.
    Returns a list of (file_path, file_info) tuples.
    """
    result = []
    
    for name, value in directory.items():
        if isinstance(value, dict):
            if "type" in value and value["type"] == "file":
                result.append((value["path"], value))
            else:
                new_prefix = f"{prefix}/{name}" if prefix else name
                result.extend(_get_all_files(value, new_prefix))
    
    return result

# Run the server
if __name__ == "__main__":
    mcp.run()
