# Research Notes Integration — Design Spec

**Date:** 2026-04-06
**Status:** Approved
**Implementation Plan:** `docs/superpowers/plans/2026-04-06-research-notes-integration.md`

---

## Problem

The blog at `calwoo.github.io` (a Hakyll 4.12 static site) has no way to surface the content in `calwoo/research-notes` — a separate public GitHub repo of structured research notes on ML, mathematics, and CS. The notes are written in Obsidian-flavored markdown with no YAML frontmatter, using wikilinks, callout blocks, and Mermaid diagrams.

## Goal

Add a new top-level "Notes" section to the blog that renders the research-notes content, keeping both repos independently maintainable while presenting a unified browsing experience.

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Content sync | Git submodule | Explicit version pinning; standard CI support; keeps the blog repo clean |
| Nav placement | New top-level section | Distinct enough from blog posts to warrant its own section; not mixed into Archive |
| Obsidian handling | Best-effort Haskell transform + preprocessing | Pandoc 2.2 Lua filter API is uncertain; Haskell AST walk is native to the codebase |
| Content depth | `note.md` only (no exercises/solutions) | Notes are public-facing; exercises/solutions can stay private in Obsidian |
| Wikilinks | Text preprocessing before Pandoc | More reliable than AST matching since Pandoc tokenization of `[[...]]` is unpredictable |
| Callouts | Pandoc AST walk on `BlockQuote` | Post-parse transform is more robust than regex on rendered HTML |
| Mermaid | Client-side via CDN script | Zero build-time complexity; Mermaid.js auto-discovers `<code class="mermaid">` blocks |

---

## Architecture

```
calwoo.github.io/
  notes/                        ← git submodule → calwoo/research-notes
    concepts/<topic>/note.md
    papers/<paper>.md
    walkthroughs/
    curricula/
  site.hs                        ← new rules + noteCompiler + noteCtx
  templates/
    note.html                    ← individual note wrapper
    notes-section.html           ← section listing (Concepts, Papers, …)
    notes-index.html             ← /notes/ landing page
    default.html                 ← +nav link, +Mermaid CDN
  css/default.css                ← +callout styles
  .gitmodules                    ← submodule config
  .circleci/config.yml           ← +submodules: true
```

**URL scheme:**
- `notes/concepts/attention-mechanisms/note.md` → `/notes/concepts/attention-mechanisms/`
- `notes/papers/muon-optimizer.md` → `/notes/papers/muon-optimizer/`
- Section index: `/notes/concepts/`
- Landing: `/notes/`

---

## Obsidian Syntax Handling

### Wikilinks
`[[target]]` → `[target](/notes/target/)` via raw text substitution before Pandoc. Targets are slugified (lowercase, spaces→hyphens). Cross-section links may produce 404s — acceptable for now.

### Callouts
`> [!NOTE] title` → `<div class="callout callout-note">...</div>` via Pandoc AST `walk`. Matches `BlockQuote [Para (Str "[!NOTE]" : …)]`. If Pandoc tokenizes `[!NOTE]` as multiple tokens, the match silently falls through to a plain blockquote.

### Mermaid
Fenced `` ```mermaid `` blocks pass through Pandoc as `<pre><code class="language-mermaid">`. Mermaid.js CDN script auto-renders them at page load.

---

## Trade-offs Not Taken

- **CI fetch (always-latest)**: Would eliminate the manual `git submodule update` step, but gives up control over when changes appear on the blog. Submodule is better for a curated publishing workflow.
- **Exercises/solutions rendering**: These exist in the notes repo but aren't published — keeps the blog focused on reference content rather than interactive exercises.
- **Full Obsidian wikilink graph**: Bidirectional links and backlinks are not implemented. Each note is self-contained.
