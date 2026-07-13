# CoordExp HTTPS Authentication

Read this only when Git needs the ignored local PAT file. Confirm first that
`github_personal_token.txt` is ignored and untracked.

Use a temporary credential helper so the token never enters remotes, commit
messages, PR bodies, command output, or tracked files:

```bash
_branch="$(git branch --show-current)"
_git_https() {
  GIT_TERMINAL_PROMPT=0 git \
    -c credential.helper= \
    -c "credential.helper=!f() { if [ \"$1\" = get ]; then echo username=x-access-token; printf 'password='; tr -d '\n' < github_personal_token.txt; echo; fi; }; f" \
    "$@"
}
_git_https fetch origin
_git_https pull --rebase origin "$_branch"
_git_https push origin HEAD
```

Skip pull when already current and push when nothing is ahead. Completion means
the intended upstream matches `HEAD` and no credential-bearing configuration or
tracked file was created.
