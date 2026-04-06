# Research Notes Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a top-level "Notes" section to the Hakyll blog that renders markdown files from the `calwoo/research-notes` GitHub repo (pulled in as a git submodule), with best-effort handling of Obsidian-flavored syntax (wikilinks, callouts, Mermaid diagrams).

**Architecture:** The research-notes repo is added as a git submodule at `notes/`. Hakyll's `site.hs` gets three new rule blocks: one for rendering individual `*.md` files under `notes/`, one for per-section index pages (Concepts, Papers, Walkthroughs, Curricula), and one for the top-level `/notes/` landing page. Obsidian wikilinks (`[[target]]`) are preprocessed as raw text substitution before Pandoc sees the file; Obsidian callouts (`> [!TYPE]`) are transformed using a Pandoc AST walk; Mermaid diagrams are rendered client-side via the Mermaid.js CDN script added to `default.html`.

**Tech Stack:** Hakyll 4.12, Pandoc 2.2, Haskell/Stack (lts-12.26), CircleCI 2.0, GitHub Pages

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `.gitmodules` | Create | Submodule config pointing to `calwoo/research-notes` |
| `notes/` | Create (submodule) | All research note content |
| `site.hs` | Modify | New Hakyll rules + helper functions for notes |
| `blog.cabal` | Modify | Add `filepath` and `text` dependencies (if not transitive) |
| `templates/note.html` | Create | Individual note page wrapper |
| `templates/notes-section.html` | Create | Per-section listing (Concepts, Papers, etc.) |
| `templates/notes-index.html` | Create | Top-level /notes/ landing page |
| `templates/default.html` | Modify | Add "Notes" nav link + Mermaid.js CDN script |
| `css/default.css` | Modify | Add callout styles |
| `.circleci/config.yml` | Modify | Enable submodule checkout |
| `docs/superpowers/specs/2026-04-06-research-notes-integration-design.md` | Create | Design spec |

---

## Task 1: Add git submodule

**Files:**
- Create: `.gitmodules`
- Create: `notes/` (submodule)

- [ ] **Step 1: Add the submodule**

```bash
cd /Users/calvinwoo/Documents/calwoo.github.io
git submodule add https://github.com/calwoo/research-notes notes
```

Expected output: Cloning into `.../notes`... done.

- [ ] **Step 2: Verify submodule structure is correct**

```bash
ls notes/
```

Expected: `concepts/  papers/  walkthroughs/  curricula/  docs/  README.md` (or similar)

- [ ] **Step 3: Check a sample note exists**

```bash
ls notes/concepts/
```

Expected: subdirectories like `attention-mechanisms/`, `category-theory/`, etc.

```bash
head -5 notes/concepts/attention-mechanisms/note.md
```

Expected: markdown content starting with `# Attention Mechanisms` or similar.

- [ ] **Step 4: Commit the submodule**

```bash
git add .gitmodules notes
git commit -m "add research-notes as git submodule"
```

---

## Task 2: Update CircleCI for submodule checkout

**Files:**
- Modify: `.circleci/config.yml:10`

The current `- checkout` step (line 10) is a plain string. Change it to a dict with `submodules: true`.

- [ ] **Step 1: Edit `.circleci/config.yml`**

Replace:
```yaml
    steps:
      - checkout
```

With:
```yaml
    steps:
      - checkout:
          submodules: true
```

- [ ] **Step 2: Verify the change looks correct**

```bash
head -15 .circleci/config.yml
```

Expected: the checkout step now has the `submodules: true` key indented beneath it.

- [ ] **Step 3: Commit**

```bash
git add .circleci/config.yml
git commit -m "enable submodule checkout in CircleCI"
```

---

## Task 3: Add helper functions to `site.hs`

**Files:**
- Modify: `site.hs`

Add the necessary imports and helper functions. These are used by the new Hakyll rules in Task 4.

- [ ] **Step 1: Add imports at the top of `site.hs`**

After the existing imports, add:

```haskell
import           Data.Char (toLower)
import           Data.List (isPrefixOf, isSuffixOf)
import           System.FilePath (dropExtension, takeFileName,
                                  takeDirectory, (</>))
import           Text.Pandoc
import           Text.Pandoc.Walk (walk)
```

- [ ] **Step 2: Add `noteRoute` function after the `postCtx` definition**

```haskell
--------------------------------------------------------------------------------
-- Route: notes/concepts/foo/note.md  → notes/concepts/foo/index.html
--        notes/papers/muon.md        → notes/papers/muon/index.html
noteRoute :: Identifier -> FilePath
noteRoute ident =
    let path       = toFilePath ident
        withoutExt = dropExtension path
        cleaned    = if takeFileName withoutExt == "note"
                     then takeDirectory withoutExt
                     else withoutExt
    in cleaned </> "index.html"
```

- [ ] **Step 3: Add `extractH1` and `stripH1` functions**

```haskell
--------------------------------------------------------------------------------
-- Extract the text of the first "# Heading" line from raw markdown.
extractH1 :: String -> String
extractH1 content =
    case filter ("# " `isPrefixOf`) (lines content) of
        (h:_) -> drop 2 h
        []    -> "Untitled"

-- Remove the first "# Heading" line so it isn't duplicated in the body
-- (default.html already renders $title$ as an <h1>).
stripH1 :: String -> String
stripH1 content =
    let ls = lines content
    in unlines $ case break ("# " `isPrefixOf`) ls of
        (before, _:after) -> before ++ after
        (before, [])      -> before
```

- [ ] **Step 4: Add `processWikilinks` function**

```haskell
--------------------------------------------------------------------------------
-- Convert Obsidian wikilinks [[target]] to standard markdown links.
-- Maps to /notes/<target>/ — Hakyll relativizeUrls handles the final path.
processWikilinks :: String -> String
processWikilinks []                    = []
processWikilinks ('[':'[':rest)        =
    case break (== ']') rest of
        (target, ']':']':after) ->
            "[" ++ target ++ "](/notes/" ++ slugify target ++ "/)"
            ++ processWikilinks after
        _ -> '[' : '[' : processWikilinks rest
processWikilinks (c:rest)              = c : processWikilinks rest

slugify :: String -> String
slugify = map (\c -> if c == ' ' then '-' else toLower c)
```

- [ ] **Step 5: Add `obsidianTransform` for callout blocks**

```haskell
--------------------------------------------------------------------------------
-- Transform Obsidian callout blockquotes into styled <div> elements.
-- Obsidian callout: > [!NOTE] optional title
-- Pandoc AST:       BlockQuote [Para (Str "[!NOTE]" : rest) : blocks]
obsidianTransform :: Pandoc -> Pandoc
obsidianTransform = walk transformBlock

transformBlock :: Block -> Block
transformBlock (BlockQuote (Para inlines : rest))
    | (firstStr : remainingInlines) <- inlines
    , Just calloutType <- parseCalloutMarker firstStr =
        Div ("", ["callout", "callout-" ++ calloutType], [])
            (Para remainingInlines : rest)
transformBlock b = b

parseCalloutMarker :: Inline -> Maybe String
parseCalloutMarker (Str s)
    | "[!" `isPrefixOf` s && "]" `isSuffixOf` s =
        Just (map toLower (drop 2 (init s)))
parseCalloutMarker _ = Nothing
```

**Note:** Pandoc 2.2's `Str` type is `Str String` (not `Text`). If you get a type error, the `Inline` constructors use `String` in this version.

- [ ] **Step 6: Add `noteCompiler` and `noteCtx`**

```haskell
--------------------------------------------------------------------------------
-- Compiler for individual notes: preprocess wikilinks, strip H1, apply
-- Pandoc with callout transform, and expose $title$ from the H1.
noteCompiler :: Compiler (Item String)
noteCompiler = do
    raw <- getResourceString
    let rawBody   = itemBody raw
        processed = processWikilinks (stripH1 rawBody)
    makeItem processed
        >>= renderPandocWith defaultHakyllReaderOptions noteWriterOptions
        >>= return . fmap (applyObsidianTransformHtml)

-- We apply obsidianTransform via the AST; but since renderPandocWith already
-- ran, we need pandocCompilerWithTransform instead. Use this version:
noteCompiler :: Compiler (Item String)
noteCompiler = do
    raw <- getResourceString
    let rawBody   = itemBody raw
        processed = processWikilinks (stripH1 rawBody)
    makeItem processed
        >>= renderPandocWithTransform
                defaultHakyllReaderOptions
                noteWriterOptions
                obsidianTransform

noteWriterOptions :: WriterOptions
noteWriterOptions =
    let mathExts    = [ Ext_tex_math_dollars
                      , Ext_tex_math_double_backslash
                      , Ext_latex_macros ]
        codeExts    = [ Ext_fenced_code_blocks
                      , Ext_backtick_code_blocks ]
        defaultExts = writerExtensions defaultHakyllWriterOptions
        newExts     = foldr enableExtension defaultExts (mathExts <> codeExts)
    in defaultHakyllWriterOptions
           { writerExtensions    = newExts
           , writerHTMLMathMethod = MathJax ""
           }

noteCtx :: Context String
noteCtx =
    field "title" (\_ -> extractH1 . itemBody <$> getResourceString)
    `mappend` defaultContext
```

**Note on `renderPandocWithTransform`:** This function exists in Hakyll 4.x as `renderPandocWithTransform :: ReaderOptions -> WriterOptions -> (Pandoc -> Pandoc) -> Item String -> Compiler (Item String)`. Verify it exists at that signature; if not, use `pandocCompilerWithTransform` instead and adjust the flow.

- [ ] **Step 7: Verify `site.hs` compiles**

```bash
stack build 2>&1 | tail -20
```

Expected: clean build. Fix any type errors before proceeding (common issues: `Str String` vs `Str Text` for the pandoc version, missing imports).

- [ ] **Step 8: Commit**

```bash
git add site.hs
git commit -m "add note helper functions to site.hs"
```

---

## Task 4: Add Hakyll rules to `site.hs`

**Files:**
- Modify: `site.hs` (inside `main`)

Add three new rule blocks inside `hakyll $ do`, before `match "templates/*"`.

- [ ] **Step 1: Add the individual note pages rule**

After the `match "posts/*"` block and before `create ["archive.html"]`, add:

```haskell
    -- Copy non-markdown assets from notes (figures, images)
    match ("notes/**" .&&. complement "notes/**.md") $ do
        route   idRoute
        compile copyFileCompiler

    -- Render individual note pages
    match "notes/**.md" $ do
        route $ customRoute noteRoute
        compile $ noteCompiler
            >>= loadAndApplyTemplate "templates/note.html"    noteCtx
            >>= loadAndApplyTemplate "templates/default.html" noteCtx
            >>= relativizeUrls
```

- [ ] **Step 2: Add per-section index pages rule**

After the notes match rule, add:

```haskell
    -- Per-section index pages
    let noteSections = ["concepts", "papers", "walkthroughs", "curricula"]
    forM_ noteSections $ \section -> do
        let sectionPattern = fromGlob ("notes/" ++ section ++ "/**.md")
        create [fromFilePath ("notes/" ++ section ++ "/index.html")] $ do
            route idRoute
            compile $ do
                notes <- loadAll sectionPattern
                let sectionCtx =
                        listField "notes" noteCtx (return notes) `mappend`
                        constField "title"   (capitalize section) `mappend`
                        constField "section" section              `mappend`
                        defaultContext
                makeItem ""
                    >>= loadAndApplyTemplate "templates/notes-section.html" sectionCtx
                    >>= loadAndApplyTemplate "templates/default.html"       sectionCtx
                    >>= relativizeUrls
```

Add the `capitalize` helper and `forM_` import:

```haskell
import           Control.Monad (forM_)
```

```haskell
capitalize :: String -> String
capitalize []     = []
capitalize (c:cs) = toLower c `seq` toUpper c : cs
  where toUpper x = if x >= 'a' && x <= 'z' then toEnum (fromEnum x - 32) else x
```

Actually simpler — replace the above `capitalize` with:
```haskell
import           Data.Char (toUpper)

capitalize :: String -> String
capitalize []     = []
capitalize (c:cs) = toUpper c : cs
```

- [ ] **Step 3: Add notes landing page rule**

```haskell
    -- Notes landing page
    create ["notes/index.html"] $ do
        route idRoute
        compile $ do
            let sections    = ["concepts", "papers", "walkthroughs", "curricula"]
                sectionsCtx =
                    listField "sections"
                        (field "name" (return . itemBody))
                        (mapM makeItem sections)
                    `mappend` constField "title" "Notes"
                    `mappend` defaultContext
            makeItem ""
                >>= loadAndApplyTemplate "templates/notes-index.html" sectionsCtx
                >>= loadAndApplyTemplate "templates/default.html"     sectionsCtx
                >>= relativizeUrls
```

- [ ] **Step 4: Verify `site.hs` compiles**

```bash
stack build 2>&1 | tail -20
```

Expected: clean build with no errors.

- [ ] **Step 5: Commit**

```bash
git add site.hs
git commit -m "add Hakyll rules for notes section"
```

---

## Task 5: Create note templates

**Files:**
- Create: `templates/note.html`
- Create: `templates/notes-section.html`
- Create: `templates/notes-index.html`

- [ ] **Step 1: Create `templates/note.html`**

```html
<article>
    <section>
        $body$
    </section>
</article>
```

*(`default.html` already renders `<h1>$title$</h1>` above `$body$`, so no title needed here.)*

- [ ] **Step 2: Create `templates/notes-section.html`**

```html
<ul>
    $for(notes)$
        <li>
            <a href="$url$">$title$</a>
        </li>
    $endfor$
</ul>
```

- [ ] **Step 3: Create `templates/notes-index.html`**

```html
<p>Research notes on machine learning, mathematics, and computer science.</p>
<ul>
    $for(sections)$
        <li>
            <a href="/notes/$name$/">$name$</a>
        </li>
    $endfor$
</ul>
```

- [ ] **Step 4: Commit**

```bash
git add templates/note.html templates/notes-section.html templates/notes-index.html
git commit -m "add note page templates"
```

---

## Task 6: Update `templates/default.html`

**Files:**
- Modify: `templates/default.html:22-25` (nav section) and `:37` (before `</body>`)

- [ ] **Step 1: Add Notes nav link**

In `templates/default.html`, change:
```html
                <a href="/archive.html">Archive</a>
```
to:
```html
                <a href="/archive.html">Archive</a>
                <a href="/notes/">Notes</a>
```

- [ ] **Step 2: Add Mermaid.js CDN script**

Before `</body>`, add:
```html
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({ startOnLoad: true });</script>
    </body>
```

- [ ] **Step 3: Commit**

```bash
git add templates/default.html
git commit -m "add Notes nav link and Mermaid.js CDN to default template"
```

---

## Task 7: Add callout styles to `css/default.css`

**Files:**
- Modify: `css/default.css` (append to end of file)

- [ ] **Step 1: Append callout styles**

Add at the end of `css/default.css`:

```css
/* Obsidian callout blocks */
.callout {
    border-left: 3px solid #aaa;
    padding: 0.5rem 1rem;
    margin: 1rem 0;
    border-radius: 0 4px 4px 0;
}
.callout-note     { border-color: #4a9eff; background: #f0f6ff; }
.callout-warning  { border-color: #f0a500; background: #fffbf0; }
.callout-tip      { border-color: #3cb371; background: #f0fff4; }
.callout-info     { border-color: #4a9eff; background: #f0f6ff; }
.callout-important { border-color: #c678dd; background: #f9f0ff; }
```

- [ ] **Step 2: Commit**

```bash
git add css/default.css
git commit -m "add callout block styles for Obsidian notes"
```

---

## Task 8: Local verification

- [ ] **Step 1: Initialize submodule if not already done**

```bash
git submodule update --init --recursive
```

- [ ] **Step 2: Build and start local server**

```bash
stack exec site watch
```

Expected: Hakyll compiles all rules and starts server at `http://localhost:8000`.
Watch for errors in the output — common issues: missing template fields, pattern match failures on notes files.

- [ ] **Step 3: Verify the Notes landing page**

Open `http://localhost:8000/notes/` in a browser.
Expected: page titled "Notes" with a list of four sections (concepts, papers, walkthroughs, curricula).

- [ ] **Step 4: Verify a section index**

Open `http://localhost:8000/notes/concepts/` in a browser.
Expected: list of concept note titles, each linking to its individual page.

- [ ] **Step 5: Verify an individual note renders**

Open one of the concept notes (e.g. `http://localhost:8000/notes/concepts/attention-mechanisms/`).
Expected: H1 title, rendered markdown body with correct math (MathJax), no raw `[[...]]` wikilink text.

- [ ] **Step 6: Verify a wikilink renders as a link**

In the rendered note, find any `[[target]]` reference.
Expected: rendered as `<a href="...">target</a>`, not literal `[[target]]`.

- [ ] **Step 7: Verify a callout renders with styling**

Find a note that uses `> [!NOTE]` callout syntax.
Expected: rendered as a styled div with left border, not a plain blockquote.
If this doesn't work, check the Pandoc AST representation by temporarily adding a `traceShow` or by running `pandoc --to native` on the note file to see how `[!NOTE]` is tokenized, then adjust `parseCalloutMarker` accordingly.

- [ ] **Step 8: Verify Mermaid renders**

Find a note that uses a `mermaid` fenced code block.
Expected: rendered as a diagram, not a code block.

- [ ] **Step 9: Verify nav bar**

Check that "Notes" appears in the nav on all pages (home, archive, individual posts, notes pages).

---

## Task 9: Write design spec and final commit

**Files:**
- Create: `docs/superpowers/specs/2026-04-06-research-notes-integration-design.md`

- [ ] **Step 1: Write design spec**

Create `docs/superpowers/specs/2026-04-06-research-notes-integration-design.md` with the content from the brainstorming session (architecture, decisions, trade-offs). Reference the plan file at `docs/superpowers/plans/2026-04-06-research-notes-integration.md`.

- [ ] **Step 2: Final commit**

```bash
git add docs/
git commit -m "add research notes integration design spec and implementation plan"
```

---

## Debugging Guide

**`stack build` type error: `Str` takes `Text` not `String`**
Pandoc 2.11+ changed `String` to `Text` in the AST. In pandoc 2.2 (used here, lts-12.26), `Str` still takes `String`. If you see this error, the LTS snapshot may have been upgraded — check `stack.yaml` and `blog.cabal`.

**Notes not showing up in section index**
Check the glob pattern in `loadAll`. Run `stack exec site build -- --verbose` to see which files Hakyll is processing. Verify the submodule is initialized (`git submodule update --init`).

**Callouts rendering as plain blockquotes**
Run `pandoc --to native notes/concepts/<any>/note.md | grep -A5 BlockQuote` to see how Pandoc tokenizes `[!NOTE]`. The first `Inline` in the `Para` may be multiple tokens rather than a single `Str "[!NOTE]"`. Adjust `parseCalloutMarker` to match the actual token sequence.

**Wikilink target not found (404)**
The `slugify` function maps spaces to hyphens and lowercases the target. Verify the note subdirectory names match the slugified wikilink targets. Cross-section wikilinks (e.g., a concept linking to a paper) will resolve to `/notes/<target>/` which may not exist — these are acceptable broken links for now.
