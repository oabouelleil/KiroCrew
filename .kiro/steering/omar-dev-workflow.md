---
inclusion: always
---

# Omar's fork workflow (dev-only, never part of a PR)

This file lives only on the `omar/dev-local` branch of the fork
(github.com/oabouelleil/KiroCrew). It is not upstream guidance.

## Remotes

- `origin`: upstream, github.com/kirodotdev/KiroCrew. Fetch only. Never push.
- `fork`: github.com/oabouelleil/KiroCrew. The only remote you push to.

## Branch layout

- `main@origin`: latest upstream main. Everything below is based on it.
- One bookmark per PR candidate (`feat/...`, `fix/...`, `omar/...`), each based on
  `main@origin`, holding only that change.
- `omar/dev-local`: dev-only files, such as this one. It adds new files only and never
  edits an upstream file, so it cannot conflict on a rebase or leak into a PR branch.
- `omar/dev`: a jj merge revision whose parents are `main@origin`, every live branch
  and `omar/dev-local`. It exists for local testing. Never open a PR from it.

## New work

Start every change on its own bookmark from main, never on `omar/dev` or
`omar/dev-local`:

```sh
jj git fetch --remote origin -b main
jj new main@origin
# ...edit, then:
jj commit -m "feat(area): what it does"
jj bookmark create feat/my-change -r @-
```

## Rebase everything onto the latest main

```sh
jj git fetch --remote origin -b main
jj git fetch --remote fork
jj rebase -b feat/my-change -d main@origin        # repeat for each live branch
jj rebase -b omar/dev-local -d main@origin
jj rebase -r omar/dev -d main@origin -d feat/my-change -d omar/dev-local   # list every live branch
jj resolve --list -r 'main@origin..omar/dev'      # must print nothing
```

Resolve a conflict in the branch that owns it (`jj new <rev>`, fix the files,
`jj squash`). Fix only cross-branch conflicts in the `omar/dev` merge itself.

## Add a branch to dev, or drop a merged one

`jj rebase -r omar/dev -d ...` replaces the merge's whole parent list, so always name
every parent you want to keep.

```sh
# add feat/new-thing
jj rebase -r omar/dev -d main@origin -d <current branches...> -d feat/new-thing -d omar/dev-local
# drop fix/merged-thing after upstream merged it
jj rebase -r omar/dev -d main@origin -d <remaining branches...> -d omar/dev-local
jj log -r 'parents(omar/dev)'                      # check the parent list
```

## Push

Push only to the fork. Check identity first, then push the bookmarks you changed:

```sh
jj config get user.email      # must print 36559888+oabouelleil@users.noreply.github.com
jj log -r 'main@origin..(omar/dev | omar/dev-local) & (author_email(glob:"*@amazon.com") | committer_email(glob:"*@amazon.com"))'
                              # must print nothing
jj git push --remote fork -b omar/dev -b omar/dev-local -b feat/my-change
```

## Rules

- Commit identity is the GitHub noreply address above. Never override it.
- Nothing pushed may carry a work email address as author or committer.
- The fork is public. Commits, branch names and files must not name internal
  company systems, hosts, paths or tools.
- Open a PR, an issue or a comment upstream only with Omar's explicit approval of
  that specific item.
