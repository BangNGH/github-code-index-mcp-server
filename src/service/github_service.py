
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
    
def git_status(repo: git.Repo) -> str:
    return repo.git.status()

def git_log(repo: git.Repo, max_count: int = 10) -> list[str]:
    commits = list(repo.iter_commits(max_count=max_count))
    log = []
    for commit in commits:
        log.append(
            f"Commit: {commit.hexsha}\n"
            f"Author: {commit.author}\n"
            f"Date: {commit.authored_datetime}\n"
            f"Message: {commit.message}\n"
        )
    return log

def git_show(repo: git.Repo, revision: str) -> str:
    commit = repo.commit(revision)
    output = [
        f"Commit: {commit.hexsha}\n"
        f"Author: {commit.author}\n"
        f"Date: {commit.authored_datetime}\n"
        f"Message: {commit.message}\n"
    ]
    if commit.parents:
        parent = commit.parents[0]
        diff = parent.diff(commit, create_patch=True)
    else:
        diff = commit.diff(git.NULL_TREE, create_patch=True)
    for d in diff:
        output.append(f"\n--- {d.a_path}\n+++ {d.b_path}\n")
        output.append(d.diff.decode('utf-8'))
    return "".join(output)