
# Import Git libraries
import os
import git

def git_clone(repo_url: str, target_path: str, branch: str | None = None) -> str:
    """Clone a git repository from a URL to a local path."""
    try:
        if os.path.exists(target_path) and os.listdir(target_path):
            return f"Error: Target directory {target_path} already exists and is not empty"
        
        clone_args = ['--progress']  # Show progress during clone
        if branch:
            clone_args.extend(['--branch', branch, '--single-branch'])
            
        repo = git.Repo.clone_from(repo_url, target_path, multi_options=clone_args)
        return f"Successfully cloned repository from {repo_url} to {target_path}"
    except git.GitCommandError as e:
        return f"Git clone error: {e}"
    except Exception as e:
        return f"Error cloning repository: {str(e)}"

def git_pull(repo_path: str) -> str:
    try:
        repo = git.Repo(repo_path)
        if repo.is_dirty():
            raise Exception("Repository has uncommitted changes")
        repo.git.pull()
        return f"Successfully pulled changes from {repo.git_dir}"
    except Exception as e:
        return f"Error pulling changes from repository: {str(e)}"