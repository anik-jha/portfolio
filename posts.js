// Dynamically load markdown posts from `public/actual_contents` and expose
// `posts` and `postsReady` on the window so the rest of the site can await it.
window.posts = [];
window.postsReady = (async () => {
  const files = [
    'quantile_regression/blog1_medium.md',
    'quantile_regression/blog2_medium.md',
    'quantile_regression/blog3_medium.md',
    'quantile_regression/blog4_medium.md',
    'quantile_regression/blog5_medium.md',
    'llm_components/llm_components_part1.md'
  ];

  // Custom snippets for each blog post
  const customSnippets = {
    'blog1-medium': 'Your model predicts the average. Great! But averages are for trivia night, not for decisions with consequences. Learn why quantile regression is essential for high-stakes ML.',
    'blog2-medium': 'Why does asymmetric loss lead to quantile predictions? Dive deep into the pinball loss function and understand the mathematical engine powering quantile regression.',
    'blog3-medium': 'Build your first quantile regression model in Python. Step-by-step implementation with statsmodels, complete with evaluation metrics and real-world examples.',
    'blog4-medium': 'Scale quantile regression to non-linear patterns with gradient boosting. Learn LightGBM and XGBoost techniques for production-grade probabilistic forecasting.',
    'blog5-medium': 'Master state-of-the-art techniques including conformal prediction, distributional regression, and neural network approaches for robust uncertainty quantification in production.',
    'llm-components-part1': 'A comprehensive guide to understanding transformers from the ground up. Learn tokenization, embeddings, positional encoding, and the architecture that powers modern LLMs like GPT and Claude.'
  };
  const pathCandidates = ['actual_contents/', 'public/actual_contents/'];

  // Showdown is included in `index.html` so it's available globally
  const converter = new showdown.Converter({
    tables: true,
    strikethrough: true,
    ghCodeBlocks: true,
    tasklists: true,
    simpleLineBreaks: false,
    openLinksInNewWindow: true
  });

  for (const file of files) {
    try {
      let res = null;
      for (const base of pathCandidates) {
        try {
          res = await fetch(base + file);
          if (res.ok) {
            break;
          }
        } catch (e) {
          // ignore and try next candidate
        }
      }
      if (!res) {
        console.warn('Could not fetch', file, 'from any path');
        continue;
      }
      if (!res.ok) {
        console.warn('Could not fetch', file, res.status);
        continue;
      }
      const text = await res.text();
      console.log('Successfully loaded:', file);

      // Check for YAML frontmatter at the top (--- ... ---) and parse it
      let title = '';
      let snippet = '';
      let date = '';
      let contentText = text;

      const fmMatch = text.match(/^---\n([\s\S]*?)\n---/);
      if (fmMatch) {
        const fm = fmMatch[1];
        // parse simple key: 'value' pairs
        fm.split(/\n/).forEach(line => {
          const m = line.match(/^(\w+):\s*['"]?(.*?)['"]?$/);
          if (m) {
            const key = m[1].trim().toLowerCase();
            const val = m[2].trim();
            if (key === 'title' && !title) title = val;
            if (key === 'date') date = val;
            if (key === 'snippet' && !snippet) snippet = val;
          }
        });
        // remove frontmatter from content string
        contentText = text.replace(fmMatch[0], '').trim();
      }

      // Extract title from first H1 (# Title) if not found in frontmatter
      if (!title) {
        const titleMatch = contentText.match(/^#\s+(.+)$/m);
        if (titleMatch) {
          title = titleMatch[1].trim();
          // Remove the title H1 from content to avoid duplication
          contentText = contentText.replace(/^#\s+(.+)$/m, '').trim();
        } else {
          title = file.replace(/\.md$/i, '');
        }
      }

      // Slug from file name (remove extension and path)
      const slug = file.split('/').pop().replace(/\.md$/i, '').replace(/_/g, '-');

      // Try to extract a date if the file contains 'Date: YYYY-MM-DD' or similar
      const dateMatch = text.match(/^(Date|date):\s*(\d{4}-\d{2}-\d{2})$/m);
      if (dateMatch) date = dateMatch[2];

      // Use custom snippet if available, otherwise extract from content
      if (customSnippets[slug]) {
        snippet = customSnippets[slug];
      } else {
        // Create a small snippet: first non-empty paragraph after the title
        // Skip italic text (*...*), horizontal rules (---), and markdown formatting
        const parts = contentText.split(/\n\n+/);
        for (const p of parts) {
          const trimmed = p.replace(/^#+\s*/, '').trim();
          if (!trimmed) continue;
          // skip the title chunk
          if (trimmed === title) continue;
          // skip italic lines starting with *
          if (trimmed.startsWith('*') && trimmed.endsWith('*')) continue;
          // skip horizontal rules
          if (trimmed.match(/^[-*_]{3,}$/)) continue;
          // skip image tags
          if (trimmed.startsWith('![')) continue;
          // skip lines that are just bold text
          if (trimmed.match(/^\*\*.*\*\*$/)) continue;
          // Get the first line and remove markdown formatting
          if (!snippet) {
            snippet = trimmed.split('\n')[0]
              .replace(/\*\*/g, '') // remove bold
              .replace(/\*/g, '')   // remove italic
              .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1') // remove links, keep text
              .replace(/`([^`]+)`/g, '$1') // remove code formatting
              .substring(0, 150); // limit length
          }
          break;
        }
      }

      // Protect math expressions before markdown conversion
      const mathPlaceholders = [];
      let mathCounter = 0;

      // Protect display math ($$...$$) FIRST - highest priority
      contentText = contentText.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
        const placeholder = `XXXXXMATHDISPLAYXXXXX${mathCounter}XXXXXMATHDISPLAYXXXXX`;
        mathPlaceholders.push({ placeholder, content: match });
        mathCounter++;
        return placeholder;
      });

      // Protect \[...\] display math
      contentText = contentText.replace(/\\\[([\s\S]*?)\\\]/g, (match) => {
        const placeholder = `XXXXXMATHDISPLAYXXXXX${mathCounter}XXXXXMATHDISPLAYXXXXX`;
        mathPlaceholders.push({ placeholder, content: match });
        mathCounter++;
        return placeholder;
      });

      // Protect \(...\) inline math - THIS IS CRITICAL
      contentText = contentText.replace(/\\\(([\s\S]*?)\\\)/g, (match) => {
        const placeholder = `XXXXXMATHINLINEXXXXX${mathCounter}XXXXXMATHINLINEXXXXX`;
        mathPlaceholders.push({ placeholder, content: match });
        mathCounter++;
        return placeholder;
      });

      // Protect inline math with LaTeX commands ($\text{...}$, $\beta$, etc) - for backward compatibility
      contentText = contentText.replace(/\$([^\$\n]*?[\\{}_^][^\$\n]*?)\$/g, (match, inner) => {
        const placeholder = `XXXXXMATHINLINEXXXXX${mathCounter}XXXXXMATHINLINEXXXXX`;
        mathPlaceholders.push({ placeholder, content: `\\(${inner}\\)` });
        mathCounter++;
        return placeholder;
      });

      // Now protect currency dollar signs (like $1.2M, $800K, $10M)
      contentText = contentText.replace(/\$(\d+(?:\.\d+)?[KMB]?)\b/g, (match, amount) => {
        // Just keep the dollar sign as-is since KaTeX won't treat single $ as delimiter anymore
        return match;
      });

      let content = converter.makeHtml(contentText);

      // Restore math expressions (replace all occurrences)
      mathPlaceholders.forEach(({ placeholder, content: mathContent }) => {
        content = content.split(placeholder).join(mathContent);
      });

      // Fix image paths: detect which series folder we're in from the file path
      const seriesFolder = file.split('/')[0]; // e.g., 'quantile_regression' or 'llm_components'
      content = content.replace(/\<img([^\>]+)src="\.\.\/assets\/([^"]+)"([^\>]*)\>/g, `<img$1src="actual_contents/${seriesFolder}/assets/$2"$3>`);
      content = content.replace(/\<img([^\>]+)src="assets\/([^"]+)"([^\>]*)\>/g, `<img$1src="actual_contents/${seriesFolder}/assets/$2"$3>`);
      // Also handle direct image references without assets/ prefix
      content = content.replace(/\<img([^\>]+)src="([^"/:]+\.(png|jpg|jpeg|gif|svg))"([^\>]*)\>/g, `<img$1src="actual_contents/${seriesFolder}/$2"$4>`);

      window.posts.push({ slug, title, date, snippet, content });
    } catch (err) {
      console.error('Error loading post', file, err);
    }
  }

  // Sort by blog number (blog1, blog2, blog3, etc.) to maintain series order
  window.posts.sort((a, b) => {
    const aMatch = a.slug.match(/blog(\d+)/);
    const bMatch = b.slug.match(/blog(\d+)/);
    if (aMatch && bMatch) {
      return parseInt(aMatch[1]) - parseInt(bMatch[1]);
    }
    return a.title.localeCompare(b.title);
  });

  console.log('Blog posts loaded successfully:', window.posts.length, 'posts');
  window.posts.forEach(p => console.log(' -', p.title, '(slug:', p.slug + ')'));
})();