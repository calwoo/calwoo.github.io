# Site Visual Polish — Design Spec

**Date:** 2026-04-08  
**Scope:** Visual/aesthetic improvements to calwoo.github.io, focused on the header and establishing a distinctive identity

---

## Goal

Add visual personality to the site without changing its minimal, academic character. Inspired by alyata.github.io's use of a logical symbol (⊢) as a branding element, the goal is to make the header feel intentional and distinctive.

## Approach

Symbol-first identity: introduce the Ω (omega) character as a logo mark in the header, paired with a warm amber accent color applied consistently across the site.

Ω was chosen because it represents the loop space in homotopy theory — a direct reference to the author's academic background in algebraic topology — and reads clearly at header size.

---

## Design

### 1. Branding — Logo Mark

The site title in the header changes from:

```
Calvin's Notebook
```

to:

```
Ω Calvin's Notebook
```

The Ω is wrapped in a `<span class="logo-symbol">` inside the existing `.logo` anchor. It is rendered in the accent color at `1.1em` size relative to the surrounding logo text, so it reads as a mark rather than just another character.

**File:** `templates/default.html`  
**Change:** Add `<span class="logo-symbol">Ω</span>` before the text content of the `.logo` anchor.

### 2. Header & Navigation

| Element | Current | New |
|---|---|---|
| Nav text transform | `uppercase` | none (lowercase) |
| Nav link hover | none (stays black) | accent color |
| Header border-bottom | `#000` (black) | accent color |
| Logo link hover | none | accent color tint |

The nav dropping `text-transform: uppercase` makes it feel more personal and less boxy. The accent color on the header border anchors the identity without adding any new structural elements.

**File:** `css/default.css`

### 3. Accent Color System

A single CSS custom property establishes the accent color:

```css
:root {
  --accent: #b85c00;
}
```

Used in exactly four places:

| Usage | Selector |
|---|---|
| Logo symbol color | `.logo-symbol` |
| Header border | `header` border-bottom |
| Nav link hover | `nav a:hover` |
| Body link color | `a` (replaces browser-default blue) |

Body links currently default to browser blue, which clashes with the serif/black aesthetic. Changing them to amber is a meaningful improvement to overall coherence.

---

## Files Modified

| File | Change |
|---|---|
| `templates/default.html` | Add `<span class="logo-symbol">Ω</span>` in logo anchor |
| `css/default.css` | Add `--accent` variable, update header border, nav hover, link color, logo-symbol styles |

---

## Out of Scope

- Homepage layout / syntomic.png replacement (potential future improvement)
- Font changes beyond nav text-transform
- Notes or post page layout changes
- Mobile-specific header behavior (inherit existing responsive breakpoints)
