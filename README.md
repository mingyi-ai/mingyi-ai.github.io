# mingyi-ai.github.io

Personal website built with [Hugo](https://gohugo.io/) and the upstream [Stack](https://github.com/CaiJimmy/hugo-theme-stack) theme.

## Local development

Clone the repository with its theme submodule:

```sh
git clone --recurse-submodules https://github.com/mingyi-ai/mingyi-ai.github.io.git
cd mingyi-ai.github.io
hugo server
```

If the repository was cloned without submodules:

```sh
git submodule update --init --recursive
```

Use Hugo Extended 0.165.0 locally to match CI. The pinned Stack release requires Hugo 0.157.0 or newer.

## Content

Posts and projects are separate Hugo sections. Both appear in the homepage feed, ordered by `lastmod`; their section pages provide filtered feeds.

Use `date` for initial publication and update `lastmod` only for meaningful public revisions. New content starts as a draft through the section archetypes:

```sh
hugo new content posts/YY-MM-DD_Title.md
hugo new content projects/project-name.md
```

## Customization

Site-specific changes live in this repository rather than in the Stack submodule:

- `assets/scss/custom.scss` — animated background integration
- `assets/js/dots-field.js` — animated background implementation
- `layouts/_partials/footer/custom.html` — background markup and fingerprinted script
- `layouts/_partials/article/components/details.html` — updated-date metadata in feed cards
- `layouts/section.html` — card-style filtered section feeds

Stack supplies article math rendering. Add `math: true` to pages that use KaTeX notation.

To update Stack, select and test an upstream release rather than tracking its default branch:

```sh
git -C themes/hugo-theme-stack fetch --tags
git -C themes/hugo-theme-stack checkout vX.Y.Z
hugo --gc --minify --panicOnWarning
git add themes/hugo-theme-stack
```

Review the local layout overrides against the corresponding upstream templates before committing a theme update.

## Deployment

Pushes to `main` are built and deployed to GitHub Pages by `.github/workflows/hugo.yaml`. Generated files under `public/` are not committed.
