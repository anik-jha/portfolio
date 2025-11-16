Project Settings: ML Engineer Portfolio

Purpose

This file, along with portfolio_plan.md and our conversation history, serves as the "memory" and context for our development process.

Project Plan

Main Plan: portfolio_plan.md

Description: This file contains the complete architecture, content strategy, and feature list for the portfolio website.

Status: Defined.

Conversation Log

Our ongoing chat conversation serves as the live log. I will refer to our previous messages to understand context, content you've provided (like resume details or blog posts), and design preferences.

Persistent Memory (Enforced)

- File: `.github/conversation_memory.md` is the authoritative, persistent memory for this project.
- Assistant Responsibilities:
	- Read this file at the start of every session to restore context.
	- After each meaningful change/decision, append a concise, dated summary.
	- If absent, create it with an initial summary and current date.
- Content Guidelines:
	- Keep entries brief and scannable (bullets preferred).
	- Capture decisions, file changes, open questions, and next steps.
	- Link to relevant files (e.g., `index.html`, `posts.js`) when helpful.

How to Use This

When you're ready to work on the site, you don't need to start from scratch. You can simply say:

"Let's start Step 1 and create the HTML skeleton."

"I'm ready to add my work experience to the resume section."

"Here is the Markdown for my first blog post. Let's add it."

"Can we change the colors on the project cards?"

I will then use the plan, the settings, and our history to generate the updated code for our single index.html file.

Memory Workflow

- Start of session: Load `.github/conversation_memory.md`, `portfolio_plan.md`, and this file.
- During work: Update `.github/conversation_memory.md` with succinct summaries after significant actions.
- End of session: Ensure outstanding next steps are recorded under a "Next Steps" bullet list.