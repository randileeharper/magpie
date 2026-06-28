# AGENTS.md

## Python environment

Use `uv` to manage the environment and run all Python commands:

```sh
uv run pytest
uv run python -m compileall magpie tests
```

Do not use system `python`, `pytest`, or package commands unless explicitly asked.

## Test commands

Preferred full verification:

```sh
uv run pytest -q
```

For focused checks:

```sh
uv run pytest tests/test_service.py -q
```

## GitHub issue/PR workflow

Use the GitHub CLI (`gh`) for issue and pull request work when available:

```sh
gh issue view <number> --json number,title,state,author,body,labels,assignees,comments,url
gh pr create --repo randileeharper/magpie --base main --head <branch> --title "..." --body "..."
gh issue close <number> --repo randileeharper/magpie --comment "Fixed by #<pr>."
```

For issue work:

1. Create a dedicated branch before editing.
2. Keep unrelated local files out of commits, especially untracked scratch directories like `tmp/`.
3. Verify with `uv run` commands before opening a PR.
4. In the PR body, include a concise summary and exact test commands run.
5. After merge, switch back to `main`, fetch/prune the remote, fast-forward local `main`, and delete the local feature branch.

This repository's GitHub remote may be named `upstream` rather than `origin`; check configured remotes if a push to `origin` fails.

## Notes

The system Python environment may not have all project dependencies installed.
