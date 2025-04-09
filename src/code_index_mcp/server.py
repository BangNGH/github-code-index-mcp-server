"""
Code Index MCP Server

This MCP server allows LLMs to index, search, and analyze code from a project directory.
It provides tools for file discovery, content retrieval, and code analysis.
"""
import datetime
import random
import string
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Tuple, Any
import os
import pathlib
import json
import fnmatch
import sys
import git
from mcp.server.fastmcp import FastMCP, Context, Image
from mcp import ServerSession, types

from service.github_service import git_clone, git_log, git_pull, git_show, git_status

# Import the ProjectSettings class - using relative import
from code_index_mcp.project_settings import ProjectSettings

DEFAULT_CLONE_PATH = os.path.join(os.getcwd(), "cloned_repos")
os.makedirs(DEFAULT_CLONE_PATH, exist_ok=True)

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

# Store indexed commits data
commit_index = {}

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


# -- Code Indexer Tools --


def git_index_commits(repo: git.Repo, max_commits: int = 100, index_file_content: bool = True) -> dict:
    """
    Index commit history for searching.
    Returns statistics about the indexing process.
    """
    global commit_index
    repo_path = repo.working_dir
    
    # Initialize index structure for this repo if it doesn't exist
    if repo_path not in commit_index:
        commit_index[repo_path] = {
            "commits": {},
            "files": {},
            "last_indexed": None
        }
    
    index_data = commit_index[repo_path]
    
    # Get commits to index
    commits = list(repo.iter_commits(max_count=max_commits))
    
    # Track stats
    stats = {
        "total_commits": len(commits),
        "indexed_commits": 0,
        "indexed_files": 0,
        "indexed_content_size_bytes": 0
    }
    
    # Process each commit
    for commit in commits:
        commit_hash = commit.hexsha
        
        # Skip if already indexed
        if commit_hash in index_data["commits"]:
            continue
            
        # Get basic commit info
        commit_info = {
            "hash": commit_hash,
            "short_hash": commit.hexsha[:7],
            "author": str(commit.author),
            "author_email": commit.author.email,
            "date": commit.authored_datetime.isoformat(),
            "message": commit.message,
            "changed_files": [],
            "file_content": {}
        }
        
        # Get file changes
        if commit.parents:
            parent = commit.parents[0]
            diffs = parent.diff(commit)
        else:
            # For initial commit, compare with empty tree
            diffs = commit.diff(git.NULL_TREE)
            
        # Process each changed file
        for diff in diffs:
            try:
                file_path = diff.b_path if diff.b_path else diff.a_path
                commit_info["changed_files"].append(file_path)
                
                # Track file history
                if file_path not in index_data["files"]:
                    index_data["files"][file_path] = []
                
                # Add commit to file history
                index_data["files"][file_path].append(commit_hash)
                
                # Index file content if flag is set and file still exists in this commit
                if index_file_content and diff.b_path:
                    try:
                        # Get file content from this commit
                        blob = commit.tree / diff.b_path
                        if blob.type == 'blob':  # Make sure it's a file, not a directory
                            file_content = blob.data_stream.read().decode('utf-8', errors='replace')
                            commit_info["file_content"][diff.b_path] = file_content
                            stats["indexed_content_size_bytes"] += len(file_content)
                            stats["indexed_files"] += 1
                    except (UnicodeDecodeError, KeyError, git.exc.GitCommandError):
                        # Skip binary files or other files that can't be decoded
                        pass
            except Exception:
                # Skip problematic diffs
                continue
                
        # Add to index
        index_data["commits"][commit_hash] = commit_info
        stats["indexed_commits"] += 1
        
    # Update last indexed timestamp
    index_data["last_indexed"] = datetime.datetime.now().isoformat()
    
    return stats

def git_search_commits(repo: git.Repo, query: str, max_results: int = 10, 
                       search_file_content: bool = True, search_commit_messages: bool = True,
                       search_file_paths: bool = True, since_date: str | None = None,
                       author: str | None = None) -> list:
    """
    Search indexed commits for matching content.
    Returns a list of matching commits with details about matches.
    """
    repo_path = repo.working_dir
    
    # Check if commits are indexed
    if repo_path not in commit_index or not commit_index[repo_path]["commits"]:
        # Index commits first
        git_index_commits(repo)
        
    index_data = commit_index[repo_path]
    results = []
    
    # Parse date filter if provided
    date_filter = None
    if since_date:
        try:
            date_filter = datetime.datetime.fromisoformat(since_date)
        except ValueError:
            # If date parsing fails, ignore the filter
            pass
    
    # Search through indexed commits
    for commit_hash, commit_info in index_data["commits"].items():
        matches = []
        
        # Apply date filter if set
        if date_filter:
            commit_date = datetime.datetime.fromisoformat(commit_info["date"])
            if commit_date < date_filter:
                continue
                
        # Apply author filter if set
        if author and author.lower() not in commit_info["author"].lower() and author.lower() not in commit_info["author_email"].lower():
            continue
            
        # Search commit message
        if search_commit_messages and query.lower() in commit_info["message"].lower():
            matches.append({
                "type": "commit_message",
                "content": commit_info["message"]
            })

        # Search file paths
        if search_file_paths:
            for file_path in commit_info["changed_files"]:
                if query.lower() in file_path.lower():
                    matches.append({
                        "type": "file_path",
                        "path": file_path
                    })

        # Search file content
        if search_file_content and "file_content" in commit_info:
            for file_path, content in commit_info["file_content"].items():
                if query.lower() in content.lower():
                    # Find matching lines
                    matching_lines = []
                    for i, line in enumerate(content.splitlines(), 1):
                        if query.lower() in line.lower():
                            matching_lines.append({
                                "line_number": i,
                                "line": line.strip()
                            })
                            
                    if matching_lines:
                        matches.append({
                            "type": "file_content",
                            "path": file_path,
                            "lines": matching_lines[:5]  # Limit to 5 matching lines per file
                        })
                        
        # Add to results if there are matches
        if matches:
            results.append({
                "commit": commit_info["short_hash"],
                "full_hash": commit_hash,
                "author": commit_info["author"],
                "date": commit_info["date"],
                "message": commit_info["message"].split("\n")[0],  # First line of message
                "matches": matches
            })
            
            # Stop if we've reached max_results
            if len(results) >= max_results:
                break
                
    return results

@mcp.tool()
def git_status_tool(repo_path: str, ctx: Context) -> str:
    """Shows the working tree status."""
    try:
        repo = git.Repo(repo_path)
        status = git_status(repo)
        return f"Project path: {repo_path}\nGit repository status:\n{status}"
    except git.InvalidGitRepositoryError:
        return f"Error: {repo_path} is not a valid Git repository."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def git_show_tool(repo_path: str, revision: str, ctx: Context) -> str:
    """Shows the contents of a commit."""
    try:
        repo = git.Repo(repo_path)
        result = git_show(repo, revision)
        return f"Project path: {repo_path}\nGit show result: {result}"
    except git.InvalidGitRepositoryError:
        return f"Error: {repo_path} is not a valid Git repository."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def git_log_tool(repo_path: str, max_count: int = 10) -> str:
    """Shows the commit logs."""
    try:
        repo = git.Repo(repo_path)
        log = git_log(repo, max_count)
        return f"Project path: {repo_path}\nCommit history:\n" + "\n".join(log)
    except git.InvalidGitRepositoryError:
        return f"Error: {repo_path} is not a valid Git repository."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def git_index_commits_tool(repo_path: str, ctx: Context, max_commits: int = 100, index_file_content: bool = True) -> str:
    """
    Index commit history for searching.
    This creates a local database of commits that can be searched efficiently.
    """
    try:
        global commit_index
        repo = git.Repo(repo_path)
        
        # Index the commits
        stats = git_index_commits(repo, max_commits, index_file_content)
        
        # Store stats in context if available
        if hasattr(ctx, 'request_context') and hasattr(ctx.request_context, 'lifespan_context'):
            ctx.request_context.lifespan_context.commit_index_stats = stats
            
            # Save commit index to disk if we have settings
            if hasattr(ctx.request_context.lifespan_context, 'settings'):
                settings = ctx.request_context.lifespan_context.settings
                try:
                    os.makedirs(settings.settings_path, exist_ok=True)
                    with open(os.path.join(settings.settings_path, "commit_index.json"), "w") as f:
                        json.dump(commit_index, f)
                except Exception as e:
                    ctx.info(f"Failed to save commit index: {e}")
        
        # Format the result message
        result = f"Project path: {repo_path}\n"
        result += f"Successfully indexed {stats['indexed_commits']} commits with {stats['indexed_files']} files.\n"
        result += f"Total indexed content: {stats['indexed_content_size_bytes'] / (1024*1024):.2f} MB"
        
        return result
    except git.InvalidGitRepositoryError:
        return f"Error: {repo_path} is not a valid Git repository."
    except Exception as e:
        return f"Error indexing commits: {str(e)}"

@mcp.tool()
def git_search_commits_tool(repo_path: str, query: str, ctx: Context, max_results: int = 10, 
                        search_file_content: bool = True, search_commit_messages: bool = True,
                        search_file_paths: bool = True, since_date: str = None,
                        author: str = None) -> Dict:
    """
    Search through indexed commits for code, messages, or file paths matching the query.
    """
    global commit_index
    
    try:

        repo = git.Repo(repo_path)
        
        # Check if commits are indexed
        if repo_path not in commit_index or "commits" not in commit_index[repo_path]:
            # Auto-index if not already indexed
            ctx.info(f"Commits not yet indexed. Indexing up to 100 commits...")
            git_index_commits(repo)
            
            # Save commit index to disk if context has settings
            if hasattr(ctx, 'request_context') and hasattr(ctx.request_context, 'lifespan_context'):
                if hasattr(ctx.request_context.lifespan_context, 'settings'):
                    settings = ctx.request_context.lifespan_context.settings
                    try:
                        os.makedirs(settings.settings_path, exist_ok=True)
                        with open(os.path.join(settings.settings_path, "commit_index.json"), "w") as f:
                            json.dump(commit_index, f)
                    except Exception as e:
                        ctx.info(f"Failed to save commit index: {e}")
        
        # Search the commits
        results = git_search_commits(
            repo, query, max_results, search_file_content, 
            search_commit_messages, search_file_paths, since_date, author
        )
        
        # Return the results
        return {
            "query": query,
            "total_results": len(results),
            "results": results,
            "project_path": repo_path,
        }
    except git.InvalidGitRepositoryError:
        return {"error": f"{repo_path} is not a valid Git repository."}
    except Exception as e:
        return {"error": f"Error searching commits: {commit_index}, {str(e)}"}


@mcp.tool()
def index_github_repo(repo_url: str, ctx: Context, branch: Optional[str] = None) -> str:
    """
    Clone a Git repository and set it as the current project path for indexing.
    """
    # Generate a random string for the target path
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    target_path = os.path.join(DEFAULT_CLONE_PATH, random_string)
    # First, clone the repository
    try:
        # Perform the clone operation without relying on git_clone_tool
        clone_result = git_clone(repo_url, target_path, branch)
        
        if "Error" in clone_result:
            return clone_result
        
        # Then set the project path to the cloned repository
        # This will correctly set up all the context for us
        set_result = set_project_path(target_path, ctx)
        
        index_result = git_index_commits_tool(target_path, ctx)
        
        return f"Project path: {target_path}\n{clone_result}\n\n{set_result}\n\n{index_result}"
    except Exception as e:
        return f"Error cloning and setting project: {str(e)}"


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
        
        # Ensure .code_indexer is added to project's .gitignore
        gitignore_path = os.path.join(abs_path, ".gitignore")
        try:
            # Check if .gitignore exists
            if os.path.exists(gitignore_path):
                # Read existing content
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if .code_indexer is already in .gitignore
                if ".code_indexer/" not in content and ".code_indexer" not in content:
                    # Append to .gitignore
                    with open(gitignore_path, 'a', encoding='utf-8') as f:
                        f.write("\n# Code Index MCP cache directory\n.code_indexer/\n")
                    ctx.info(f"Added .code_indexer/ to project's .gitignore file.")
            else:
                # Create new .gitignore
                with open(gitignore_path, 'w', encoding='utf-8') as f:
                    f.write("# Code Index MCP cache directory\n.code_indexer/\n")
                ctx.info(f"Created .gitignore file with .code_indexer/ entry.")
        except Exception as gitignore_error:
            ctx.info(f"Note: Could not update .gitignore file: {gitignore_error}")
        
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
        
        return f"Project path set to: `{abs_path}`. Indexed {file_count} files in the project."
    except Exception as e:
        return f"Error setting project path: {e}"

@mcp.tool()
def search_code(query: str, project_path: str, ctx: Context, extensions: Optional[List[str]] = None, case_sensitive: bool = False) -> Dict[str, List[Tuple[int, str]]]:
    """
    Search for code matches within the indexed files.
    Returns a dictionary mapping filenames to lists of (line_number, line_content) tuples.
    """
    project_path = os.path.normpath(project_path)
    if project_path.startswith('..'):
        return {"error": f"Invalid file path: {project_path}"}

    base_path = ctx.request_context.lifespan_context.base_path if ctx.request_context.lifespan_context.base_path else project_path
    
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
            if False:
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
    if len(results) > 0:
        results["Project path"] = base_path
    return results

@mcp.tool()
def find_files(pattern: str, project_path: str, ctx: Context) -> List[str]:
    """
    Find files in the project that match the given pattern.
    Supports glob patterns like *.py or **/*.js.
    """
    project_path = os.path.normpath(project_path)
    if project_path.startswith('..'):
        return {"error": f"Invalid file path: {project_path}"}
    base_path = ctx.request_context.lifespan_context.base_path if ctx.request_context.lifespan_context.base_path else project_path
    
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
    if len(matching_files) > 0:
        matching_files.append(f"Project path: {base_path}") 
    return matching_files

@mcp.tool()
def get_file_summary(file_path: str, project_path: str, ctx: Context, include_file_content: bool = False) -> Dict[str, Any]:
    """
    Get a summary of a specific file, including:
    - Line count
    - Function/class definitions (for supported languages)
    - Import statements
    - Basic complexity metrics
    """
    project_path = os.path.normpath(project_path)
    if project_path.startswith('..'):
        return {"error": f"Invalid file path: {project_path}"}
    base_path = ctx.request_context.lifespan_context.base_path if ctx.request_context.lifespan_context.base_path else project_path
    
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
        if False:
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
            "project_path": base_path,
            "file_path": norm_path,
            "line_count": line_count,
            "size_bytes": os.path.getsize(full_path),
            "extension": ext,
        }
        if include_file_content:
            summary["file_content"] = content
        
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
def refresh_index(project_path: str, ctx: Context) -> str:
    """Refresh the project index."""
    project_path = os.path.normpath(project_path)
    if project_path.startswith('..'):
        return {"error": f"Invalid file path: {project_path}"}

    base_path = ctx.request_context.lifespan_context.base_path if ctx.request_context.lifespan_context.base_path else project_path

    # Check if base_path is set
    if not base_path:
        return "Error: Project path not set. Please use set_project_path to set a project directory first."
    
    # Clear existing index
    global file_index
    file_index.clear()
    
    # Pull for latest changes
    try:
        git_pull(base_path)
    except Exception as e:
        return f"Failed to pull latest changes. Project re-index failed. Project path: {base_path}. Error: {e}"
    
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
    
    return f"Project path: `{base_path}` re-indexed. Found {file_count} files."

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
        - Windows: "C:/Users/project_path/projects/my-project"
        - macOS/Linux: "/home/project_path/projects/my-project"
        
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

def main():
    """Entry point for the code indexer."""
    print("Starting Code Index MCP Server...", file=sys.stderr)
    mcp.run(transport="sse")

if __name__ == "__main__":
    main()
