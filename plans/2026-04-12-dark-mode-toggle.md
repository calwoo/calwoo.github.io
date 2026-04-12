# Dark Mode Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sun/moon icon toggle in the nav bar that switches the site between light and dark themes, respects OS preference by default, and persists the user's choice via `localStorage`.

**Architecture:** CSS custom properties already defined in `:root` are overridden by a `[data-theme="dark"]` attribute on `<html>`. A tiny inline script in `<head>` sets this attribute before first paint (preventing any flash). A click handler in the nav toggle flips the attribute and writes to `localStorage`.

**Tech Stack:** Vanilla HTML/CSS/JS. No build tools or dependencies. Site is Hakyll-generated but only template and CSS files are touched — no Haskell changes required.

---

### Task 1: Add dark mode CSS overrides to `css/default.css`

**Files:**
- Modify: `css/default.css`

- [ ] **Step 1: Append dark mode overrides to end of `css/default.css`**

Add the following block at the very end of the file:

```css
/* ── Dark mode ─────────────────────────────────── */

[data-theme="dark"] {
  --card-bg:     #1e1c19;
  --card-bg-alt: #2a2723;
}

[data-theme="dark"] body {
  color: #d4cfc8;
}

[data-theme="dark"] nav a {
  color: #d4cfc8;
}

[data-theme="dark"] .logo a {
  color: #d4cfc8;
}

[data-theme="dark"] footer {
  color: #666;
  border-color: #444;
}

[data-theme="dark"] article .header {
  color: #888;
}

[data-theme="dark"] th,
[data-theme="dark"] td {
  border-color: #444;
}

[data-theme="dark"] thead th,
[data-theme="dark"] tbody tr:last-child td {
  border-color: #888;
}

[data-theme="dark"] .callout-note,
[data-theme="dark"] .callout-info      { background: #1a2535; }
[data-theme="dark"] .callout-warning   { background: #2a2010; }
[data-theme="dark"] .callout-tip       { background: #102018; }
[data-theme="dark"] .callout-important { background: #251530; }

/* Toggle button — matches existing .nav-github style */
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

- [ ] **Step 2: Verify by manually opening `_site/index.html` in a browser**

Build the site first:
```bash
stack exec site -- build
```
Open `_site/index.html` in a browser. Run this in the browser console to check the override works:
```js
document.documentElement.setAttribute('data-theme', 'dark')
```
Expected: card background turns dark warm brown, text turns light, accent color stays orange.

Run again to remove:
```js
document.documentElement.removeAttribute('data-theme')
```
Expected: returns to light cream card.

- [ ] **Step 3: Commit**

```bash
git add css/default.css
git commit -m "style: add dark mode CSS overrides via [data-theme=dark]"
```

---

### Task 2: Add dark syntax highlighting override to `css/syntax.css`

**Files:**
- Modify: `css/syntax.css`

- [ ] **Step 1: Append dark code block override at end of `css/syntax.css`**

```css
/* ── Dark mode ─────────────────────────────────── */

[data-theme="dark"] pre.sourceCode {
  background-color: #2a2723;
  border-color: #444;
}
```

- [ ] **Step 2: Verify manually**

With the site already built (`stack exec site -- build`), open a post that has code blocks, e.g. `_site/posts/2020-01-21-profunctors.html`.

Run in browser console:
```js
document.documentElement.setAttribute('data-theme', 'dark')
```
Expected: code blocks shift from light grey to dark warm brown, border dims.

- [ ] **Step 3: Commit**

```bash
git add css/syntax.css
git commit -m "style: dark mode override for syntax-highlighted code blocks"
```

---

### Task 3: Add init script, toggle button, and click handler to `templates/default.html`

**Files:**
- Modify: `templates/default.html`

- [ ] **Step 1: Add inline init script in `<head>`, before the first `<link>` stylesheet**

In `templates/default.html`, find:
```html
        <link rel="icon" type="image/x-icon" href="/favicon.ico" />
        <meta charset="utf-8">
        <meta http-equiv="x-ua-compatible" content="ie=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>$title$ - Calvin Woo's blog</title>
        <link rel="stylesheet" href="/css/default.css" />
```

Replace with:
```html
        <link rel="icon" type="image/x-icon" href="/favicon.ico" />
        <meta charset="utf-8">
        <meta http-equiv="x-ua-compatible" content="ie=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>$title$ - Calvin Woo's blog</title>
        <script>
            (function() {
                var saved = localStorage.getItem('theme');
                var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                if (saved === 'dark' || (!saved && prefersDark)) {
                    document.documentElement.setAttribute('data-theme', 'dark');
                }
            })();
        </script>
        <link rel="stylesheet" href="/css/default.css" />
```

The script must appear before the stylesheet links. The browser parses and runs it synchronously, setting `data-theme` on `<html>` before any CSS is applied — preventing a flash of the wrong theme.

- [ ] **Step 2: Add toggle button to `<nav>`**

Find the closing `</nav>` tag in `templates/default.html`:
```html
                <a href="https://github.com/calwoo" aria-label="GitHub" class="nav-github">
                    <svg height="1.6rem" viewBox="0 0 16 16" version="1.1" width="1.6rem" aria-hidden="true">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                    </svg>
                </a>
            </nav>
```

Replace with:
```html
                <a href="https://github.com/calwoo" aria-label="GitHub" class="nav-github">
                    <svg height="1.6rem" viewBox="0 0 16 16" version="1.1" width="1.6rem" aria-hidden="true">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                    </svg>
                </a>
                <button id="theme-toggle" aria-label="Toggle dark mode" class="nav-theme-toggle"></button>
            </nav>
```

- [ ] **Step 3: Add click-handler script before `</body>`**

Find the last two lines before `</body>` in `templates/default.html`:
```html
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({ startOnLoad: true });</script>
    </body>
```

Replace with:
```html
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({ startOnLoad: true });</script>
        <script>
            (function() {
                var MOON = '<svg xmlns="http://www.w3.org/2000/svg" height="1.6rem" width="1.6rem" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
                var SUN  = '<svg xmlns="http://www.w3.org/2000/svg" height="1.6rem" width="1.6rem" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';

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
    </body>
```

Icon logic: the moon icon is shown when the site is in **light** mode (click to go dark). The sun icon is shown when in **dark** mode (click to go light). Both SVGs use `1.6rem` dimensions to match the GitHub nav icon.

- [ ] **Step 4: Rebuild and verify end-to-end**

```bash
stack exec site -- build
```

Open `_site/index.html` in a browser. Check:

1. If your OS is in dark mode: page loads with dark card — no flash of light theme.
2. If your OS is in light mode: page loads with light card.
3. Click the toggle icon — theme flips, icon swaps between sun and moon.
4. Reload the page — theme stays as you left it (localStorage persisted).
5. Open browser DevTools → Application → Local Storage → check `theme` key is set to `"dark"` or `"light"`.
6. Run `localStorage.removeItem('theme')` in console, reload — OS preference is respected again.
7. Nav looks correct on mobile (< 640px): toggle icon appears in centered nav row.

- [ ] **Step 5: Commit**

```bash
git add templates/default.html
git commit -m "feat: add dark mode toggle to nav bar"
```
