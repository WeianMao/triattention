# Development Guide

Internal reference for anyone picking up this work. Keep this file up to date.

---

## 1. Repository Layout

```
/data/rbg/users/weian/project/rl/
├── dc1/              — Private repo, main branch. Original codebase (R-KV structure).
│                       Used for: experiments, internal scripts, release docs, archive.
│                       Remote: git@github.com:WeianMao/dc.git
│
├── dc1-release/      — Private repo, release/public branch (worktree of dc1).
│                       Same directory structure as the public repo.
│                       Used for: active development, testing before publishing.
│                       This is the "bridge" between private and public.
│
└── triattention/     — Public repo. Clean, release-ready code only.
                        Remote: git@github.com:WeianMao/triattention.git
                        ArXiv: https://arxiv.org/abs/2604.04921
                        Project: https://weianmao.github.io/tri-attention-project-page/
```

---

## 2. Development Workflow

- **Active development happens in `dc1-release/`** (the release/public branch).
- When ready to publish: manually copy clean changes to `triattention/`, review the diff, then push.
- When a teammate pushes to the public repo: pull in `triattention/`, then cherry-pick relevant commits into `dc1-release/`.
- **`dc1/` main branch is archive-only** — no active development happens there.

---

## 3. Code Isolation Rules

- **NEVER push `dc1-release/` to the public repo directly.** It may contain internal paths, configs, or defaults not suitable for public release.
- Internal configs, experiment data, and this `internal_docs/` folder must NOT be copied to `triattention/`.
- Before copying anything to the public repo, scan for:
  - Internal paths (e.g., `/data/rbg/`)
  - Usernames (e.g., `weian`)
  - Experiment configs and internal-only defaults
- The `internal_docs/` directory is private — never include it in public releases.

---

## 4. Syncing Changes

| Direction | Method |
|---|---|
| Public → Release branch | `git fetch public && git cherry-pick <commits>` |
| Release branch → Public | Manually copy files to `triattention/`, review diff, commit and push |

**Never use `git merge` between the two repos.** Use cherry-pick or manual copy only to keep histories clean and avoid leaking internal commits.

---

## 5. Key Reminders

- `.gitignore` covers experiment outputs, logs, and internal configs — verify it is up to date when adding new artifact types.
- The release branch may have development-convenience defaults that differ from public defaults. Audit these before publishing.
- Always run a sensitive content scan before pushing to the public repo.
