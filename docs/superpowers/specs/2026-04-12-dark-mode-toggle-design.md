# Dark Mode Toggle — Design Spec

**Date:** 2026-04-12
**Status:** Approved

---

## Overview

Add a dark mode toggle to the Hakyll static blog. The toggle lives in the nav bar as a sun/moon icon button. It respects the user's OS preference by default and persists manual overrides via `localStorage`.

---

## Architecture

The implementation is entirely client-side (no server changes, no Hakyll Haskell changes beyond template edits). It touches three files:

| File | Change |
|---|---|
| `templates/default.html` | Inline init script in `<head>`, toggle button in `<nav>`, click-handler script before `</body>` |
| `css/default.css` | `[data-theme="dark"]` overrides for CSS custom properties and element colors |
| `css/syntax.css` | Dark variant for `pre.sourceCode` code block background |

---

## CSS Changes (`css/default.css`)

The existing `:root` block defines light-mode variables. Dark mode is applied via a `[data-theme="dark"]` attribute on `<html>`, overriding only the values that change:

```css
[data-theme="dark"] {
  --card-bg:     #1e1c19;
  --card-bg-alt: #2a2723;
}

/* Body text */
[data-theme="dark"] body {
  color: #d4cfc8;
}

/* Nav links */
[data-theme="dark"] nav a {
  color: #d4cfc8;
}

/* Logo */
[data-theme="dark"] .logo a {
  color: #d4cfc8;
}

/* Footer */
[data-theme="dark"] footer {
  color: #666;
  border-color: #444;
}

/* Article metadata */
[data-theme="dark"] article .header {
  color: #888;
}

/* Table borders */
[data-theme="dark"] th,
[data-theme="dark"] td {
  border-color: #444;
}
[data-theme="dark"] thead th,
[data-theme="dark"] tbody tr:last-child td {
  border-color: #888;
}

/* Callout blocks */
[data-theme="dark"] .callout-note,
[data-theme="dark"] .callout-info     { background: #1a2535; }
[data-theme="dark"] .callout-warning  { background: #2a2010; }
[data-theme="dark"] .callout-tip      { background: #102018; }
[data-theme="dark"] .callout-important { background: #251530; }

/* Toggle button */
.nav-theme-toggle {
  background: none;
  border: none;
  padding: 0;
  cursor: pointer;
  color: inherit;
  display: flex;
  align-items: center;
}
.nav-theme-toggle svg {
  fill: currentColor;
  display: block;
}
```

The outer page background (`--page-bg: #2e2b28`) is intentionally left unchanged — it is already dark in both modes, providing the framing effect.

## CSS Changes (`css/syntax.css`)

Code block background flips from light grey to a warm dark:

```css
[data-theme="dark"] pre.sourceCode {
  background-color: #2a2723;
  border-color: #444;
}
```

---

## Template Changes (`templates/default.html`)

### 1. Inline init script in `<head>` (before stylesheets)

Placed as the very first `<script>` tag, before any `<link>` stylesheets, so the attribute is set before the browser paints:

```html
<script>
  (function() {
    var saved = localStorage.getItem('theme');
    var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (saved === 'dark' || (!saved && prefersDark)) {
      document.documentElement.setAttribute('data-theme', 'dark');
    }
  })();
</script>
```

### 2. Toggle button in `<nav>`

Added as the last item in `<nav>`, after the GitHub icon, using the same visual pattern (SVG icon, no label):

```html
<button id="theme-toggle" aria-label="Toggle dark mode" class="nav-theme-toggle">
  <!-- icon set by JS on load -->
</button>
```

### 3. Click-handler script before `</body>`

Placed after the existing Mermaid scripts, wires up the toggle and sets the correct initial icon:

```html
<script>
  (function() {
    var SUN = '<svg ...></svg>';   /* sun icon SVG */
    var MOON = '<svg ...></svg>';  /* moon icon SVG */
    var btn = document.getElementById('theme-toggle');

    function isDark() {
      return document.documentElement.getAttribute('data-theme') === 'dark';
    }
    function setIcon() {
      btn.innerHTML = isDark() ? SUN : MOON;
    }

    setIcon();
    btn.addEventListener('click', function() {
      var dark = !isDark();
      document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
      localStorage.setItem('theme', dark ? 'dark' : 'light');
      setIcon();
    });
  })();
</script>
```

Sun and moon SVGs are inline Heroicons-style paths at `1.6rem` width/height to match the GitHub icon already in the nav.

---

## Data Flow

```
Page load
  └── inline <head> script
        ├── reads localStorage('theme')
        ├── falls back to prefers-color-scheme
        └── sets data-theme on <html>  ← CSS picks this up immediately

User clicks toggle
  ├── flips data-theme on <html>       ← CSS updates instantly
  ├── writes to localStorage           ← persists across sessions
  └── swaps icon (sun ↔ moon)
```

---

## Constraints & Non-Goals

- No changes to Hakyll's Haskell build code (`site.hs`).
- KaTeX/MathJax and TikZJax SVGs are not explicitly dark-mode styled — they render well enough on the dark card background in practice.
- Mermaid diagrams use their own theming; not in scope for this change.
- No server-side cookie or HTTP header — purely `localStorage`.
