# Agent working agreement

These instructions apply to every agent and subagent working in this repository.

## Boy Scout rule

Leave the repository and worktree at least as understandable and operable as you
found them. This is a hygiene rule, not permission for unrelated refactors.

- Improve issues directly encountered within the assigned scope when the fix is
  small, low-risk, and verifiable.
- Do not perform speculative cleanup, mass formatting, dependency upgrades, or
  architectural rewrites under the Boy Scout rule.
- Preserve all pre-existing changes. A dirty worktree may contain user work; never
  discard, overwrite, stash, or reformat it merely to obtain a clean status.
- If unrelated damage or clutter cannot be safely corrected, identify it clearly in
  the handoff instead of silently changing it.

## Start-of-work audit

Before editing, inspect and record:

1. `git status -sb`
2. the current branch and its relation to the intended base
3. `git worktree list`
4. existing modified and untracked files relevant to the task

Read the repository documentation and inspect the implementation before proposing
architecture. If files change unexpectedly during the task, stop and determine
whether another agent is working in the same tree before continuing.

## Parallel-agent isolation

- Agents doing parallel implementation must use separate Git worktrees and branches.
- Do not run two editing agents in the same worktree.
- Give each agent concrete file or component ownership; avoid overlapping write
  scopes. Read-only agents may share a checkout.
- Create worktrees from an explicit commit or remote branch, and report their paths
  and branches in the handoff.
- Do not move another agent's uncommitted work between worktrees without explicit
  coordination.

## Editing and generated files

- Keep changes narrowly tied to the assigned outcome.
- Follow existing terminology, formatting, and project patterns.
- Put build trees, caches, logs, generated evidence, and temporary files under an
  ignored path or `/tmp`. Verify unfamiliar paths with `git check-ignore` before
  generating large outputs.
- Remove only artifacts created by the current task. Never delete an untracked file
  merely because its owner is unknown.
- Do not add broad ignore patterns solely to hide accidental task output.

## Testing and review

- Work test-first for behavioral changes: establish a failing test, implement the
  smallest coherent fix, then run focused and broader checks proportional to risk.
- Run formatting/linting and `git diff --check` before handoff.
- Review the final diff for accidental generated files, debug code, secrets, absolute
  local paths, and unrelated edits.
- Report commands and observed results accurately. Distinguish verified facts from
  estimates, historical evidence, and checks that could not be run.

## Commits, pushes, and handoff

- Do not commit or push unless the user explicitly requests it.
- In a mixed worktree, stage explicit paths only. Do not use `git add -A` or
  `git add .`.
- Before committing, inspect `git diff --cached --check`, the staged file list, and
  the staged diff/stat.
- Never use destructive cleanup commands such as `git reset --hard`, `git clean`, or
  `git checkout --` without explicit user authorization.
- At handoff, include the branch, commit/base, checks run, remaining dirty files,
  generated artifacts, and any worktrees that should be retained or removed.

"Better shape" does not necessarily mean "clean." Preserving and clearly accounting
for legitimate in-progress work is better than deleting it.

## WSL2 and CUDA safety

- Limit CUDA builds to `-j4`; do not run heavy Docker workloads alongside them.
- Do not start the `cyclos-app` container during CUDA builds.
- If a CUDA build triggers OOM and the user system bus remains unavailable, use
  `dbus-run-session -- codex`; with administrator access, restarting
  `user@1000.service` restores the user manager and session bus.
