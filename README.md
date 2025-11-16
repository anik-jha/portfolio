This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Content and blog posts

- Put blog posts as Markdown files in `public/actual_contents` (or `actual_contents/` during local development). The site will dynamically load the `blog*.md` files and render them on the Blog page.
- To add a new post, add a new `blogX.md` file and include an optional YAML frontmatter block for `title`, `date`, and `snippet`.

Example frontmatter:

```md
---
title: "My New Post"
date: 2025-11-01
snippet: "Short summary of the post."
---

# My New Post

Content goes here.
```

The resume is loaded from `public/actual_contents/resume.md` if available (or falls back to `public/actual_contents/sample_resume.md`), so you can replace that file with your real resume to show it on the Experience page.

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
