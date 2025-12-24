// Dynamically load markdown posts from `public/actual_contents` and expose
// `posts` and `postsReady` on the window so the rest of the site can await it.
window.posts = [];
window.postsReady = (async () => {
  const files = [
    'xAI/data_attribution_part1.md',
    // 'xAI/data_attribution_part2.md',  // Hidden for now
    'quantile_regression/blog1_medium.md',
    'quantile_regression/blog2_medium.md',
    'quantile_regression/blog3_medium.md',
    'quantile_regression/blog4_medium.md',
    'quantile_regression/blog5_medium.md',
    'llm_components/llm_components_part1.md',
    'llm_components/llm_components_part2.md',
    'pet_projects/daily_paper_reader/daily-paper-reader.md',
    'NeurIPS2025/neurips2025_tutorials_report.md'
  ];

  // Custom snippets for each blog post
  const customSnippets = {
    'data-attribution-part1': 'When your LLM says 2+2=5, you need to find the smoking gun. Learn gradient alignment, influence functions, and how to build attribution pipelines that trace model behavior back to the exact training rows that broke it.',
    'data-attribution-part2': 'Prove your attribution was right by actually removing the data and retraining (the scientific method for ML). Then scale to billions of rows using TRAK, DataInf, and production tricks that don\'t require a PhD in infrastructure.',
    'blog1-medium': 'Your model predicts the average. Great! But averages are for trivia night, not for decisions with consequences. Learn why quantile regression is essential for high-stakes ML.',
    'blog2-medium': 'Why does asymmetric loss lead to quantile predictions? Dive deep into the pinball loss function and understand the mathematical engine powering quantile regression.',
    'blog3-medium': 'Build your first quantile regression model in Python. Step-by-step implementation with statsmodels, complete with evaluation metrics and real-world examples.',
    'blog4-medium': 'Scale quantile regression to non-linear patterns with gradient boosting. Learn LightGBM and XGBoost techniques for production-grade probabilistic forecasting.',
    'blog5-medium': 'Master state-of-the-art techniques including conformal prediction, distributional regression, and neural network approaches for robust uncertainty quantification in production.',
    'llm-components-part1': 'A comprehensive guide to understanding transformers from the ground up. Learn tokenization, embeddings, positional encoding, and the architecture that powers modern LLMs like GPT and Claude.',
    'llm-components-part2': 'Dive deep into the attention mechanism—the revolutionary innovation that changed AI forever. Explore self-attention, multi-head attention, and feed-forward networks with intuitive explanations and complete Python implementations.',
    'daily-paper-reader': 'I built a tool to help me keep up with the flood of AI papers. It fetches the latest papers from ArXiv and OpenReview, summarizes them using a LLM, and presents them in a clean, daily digest.',
    'neurips2025-tutorials-report': 'Deep dive into six game-changing tutorials from NeurIPS 2025: XAI methods, benchmarking best practices, autoregressive models, imitation learning, and model merging. Technical insights with a side of humor from what 15,000 ML researchers learned in San Diego.'
  };
  const pathCandidates = ['actual_contents/', 'public/actual_contents/'];

  // Showdown is included in `index.html` so it's available globally
  const converter = new showdown.Converter({
    tables: true,
    strikethrough: true,
    ghCodeBlocks: true,
    tasklists: true,
    simpleLineBreaks: false,
    openLinksInNewWindow: true,
    emoji: true,
    underline: true,
    completeHTMLDocument: false,
    encodeEmails: true,
    ghCompatibleHeaderId: true,
    headerLevelStart: 1,
    literalMidWordUnderscores: true,
    parseImgDimensions: true,
    simplifiedAutoLink: true,
    excludeTrailingPunctuationFromURLs: true,
    ghMentions: false,
    smoothLivePreview: false,
    prefixHeaderId: false,
    disableForced4SpacesIndentedSublists: false,
    backslashEscapesHTMLTags: false,
    tablesHeaderId: false
  });

  for (const file of files) {
    try {
      let res = null;
      for (const base of pathCandidates) {
        try {
          res = await fetch(base + file, { cache: "no-store" });
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

      // Convert ALL $...$ inline math to \(...\) format (excluding currency)
      // First protect currency (e.g., $1.2M, $800K, $10M)
      const currencyPlaceholders = [];
      let currencyCounter = 0;
      contentText = contentText.replace(/\$(\d+(?:\.\d+)?[KMB]?)\b/g, (match) => {
        const placeholder = `XXXXXCURRENCYXXXXX${currencyCounter}XXXXXCURRENCYXXXXX`;
        currencyPlaceholders.push({ placeholder, content: match });
        currencyCounter++;
        return placeholder;
      });

      // Now convert all remaining $...$ to \(...\) and protect
      contentText = contentText.replace(/\$([^\$\n]+?)\$/g, (match, inner) => {
        const placeholder = `XXXXXMATHINLINEXXXXX${mathCounter}XXXXXMATHINLINEXXXXX`;
        mathPlaceholders.push({ placeholder, content: `\\(${inner}\\)` });
        mathCounter++;
        return placeholder;
      });

      // Restore currency placeholders
      currencyPlaceholders.forEach(({ placeholder, content: currencyContent }) => {
        contentText = contentText.split(placeholder).join(currencyContent);
      });

      let content = converter.makeHtml(contentText);

      // Restore math expressions (replace all occurrences)
      mathPlaceholders.forEach(({ placeholder, content: mathContent }) => {
        content = content.split(placeholder).join(mathContent);
      });

      // Fix image paths: use the file's directory as the base
      const fileDirectory = file.substring(0, file.lastIndexOf('/'));

      content = content.replace(/\<img([^\>]+)src="\.\.\/assets\/([^"]+)"([^\>]*)\>/g, `<img$1src="actual_contents/${fileDirectory}/assets/$2"$3>`);
      content = content.replace(/\<img([^\>]+)src="assets\/([^"]+)"([^\>]*)\>/g, `<img$1src="actual_contents/${fileDirectory}/assets/$2"$3>`);
      // Also handle direct image references without assets/ prefix
      content = content.replace(/\<img([^\>]+)src="([^"/:]+\.(png|jpg|jpeg|gif|svg))"([^\>]*)\>/g, `<img$1src="actual_contents/${fileDirectory}/$2"$4>`);

      // Determine series from folder path
      let series = 'other';
      if (file.includes('quantile_regression')) series = 'quantile_regression';
      else if (file.includes('llm_components')) series = 'llm_components';
      else if (file.includes('pet_projects')) series = 'pet_projects';
      else if (file.includes('NeurIPS2025')) series = 'neurips2025';
      else if (file.includes('xAI')) series = 'xai';

      window.posts.push({ slug, title, date, snippet, content, series });
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