# mingyi-ai.github.io

Personal website built with [Hugo](https://gohugo.io/) and the upstream [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme.

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

Use Hugo Extended 0.165.0 locally to match CI. PaperMod currently requires Hugo 0.146.0 or newer.

## Customization

Site-specific changes live in this repository rather than in a PaperMod fork:

- `assets/css/extended/custom.css` — profile, post, and background styles
- `layouts/_partials/extend_head.html` — KaTeX setup
- `layouts/_partials/extend_footer.html` — animated background markup and script
- `static/js/dots-field.js` — animated background implementation

To update PaperMod:

```sh
git submodule update --remote themes/PaperMod
git add themes/PaperMod
git commit -m "Update PaperMod"
```

Review and build the site before committing a theme update.

## Deployment

Pushes to `main` are built and deployed to GitHub Pages by `.github/workflows/hugo.yaml`. Generated files under `public/` are not committed.
