{-# LANGUAGE OverloadedStrings #-}

import           Data.Monoid (mappend)
import           Hakyll
import           Text.Pandoc.Options

import           Data.Char (toLower)
import           Data.List (isPrefixOf, isSuffixOf)
import           System.FilePath (dropExtension, (</>))
import           Text.Pandoc (Block(..), Inline(..), Pandoc)
import           Text.Pandoc.Walk (walk)


main :: IO ()
main = hakyll $ do
    match "images/*" $ do
        route   idRoute
        compile copyFileCompiler

    --- add the favicon
    match "favicon.ico" $ do
        route   idRoute
        compile copyFileCompiler

    match "css/*" $ do
        route   idRoute
        compile compressCssCompiler

    match (fromList ["about.rst", "contact.markdown"]) $ do
        route   $ setExtension "html"
        compile $ pandocCompiler_
            >>= loadAndApplyTemplate "templates/default.html" defaultContext
            >>= relativizeUrls

    match "posts/*" $ do
        route $ setExtension "html"
        compile $ pandocCompiler_
            >>= loadAndApplyTemplate "templates/post.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx
            >>= relativizeUrls

    create ["archive.html"] $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            let archiveCtx =
                    listField "posts" postCtx (return posts) `mappend`
                    constField "title" "Archives"            `mappend`
                    defaultContext

            makeItem ""
                >>= loadAndApplyTemplate "templates/archive.html" archiveCtx
                >>= loadAndApplyTemplate "templates/default.html" archiveCtx
                >>= relativizeUrls


    match "index.html" $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            let indexCtx =
                    listField "posts" postCtx (return posts) `mappend`
                    constField "title" "Home"                `mappend`
                    defaultContext

            getResourceBody
                >>= applyAsTemplate indexCtx
                >>= loadAndApplyTemplate "templates/default.html" indexCtx
                >>= relativizeUrls

    match "templates/*" $ compile templateBodyCompiler


--------------------------------------------------------------------------------
postCtx :: Context String
postCtx =
    dateField "date" "%B %e, %Y" `mappend`
    defaultContext

--------------------------------------------------------------------------------
-- custom pandoc compiler for LaTeX support

pandocCompiler_ :: Compiler (Item String)
pandocCompiler_ =
    let mathExts =
            [ Ext_tex_math_dollars
            , Ext_tex_math_double_backslash
            , Ext_latex_macros
            ]
        codeExts =
            [ Ext_fenced_code_blocks
            , Ext_backtick_code_blocks
            ]
        defaultExts = writerExtensions defaultHakyllWriterOptions
        newExts = foldr enableExtension defaultExts (mathExts <> codeExts)
        writerOptions =
            defaultHakyllWriterOptions {
                writerExtensions = newExts
              , writerHTMLMathMethod = MathJax ""
            }
    in pandocCompilerWith defaultHakyllReaderOptions writerOptions

--------------------------------------------------------------------------------
-- Route: notes/concepts/foo/bar.md  → notes/concepts/foo/bar/index.html
--        notes/papers/muon.md       → notes/papers/muon/index.html
noteRoute :: Identifier -> FilePath
noteRoute ident = dropExtension (toFilePath ident) </> "index.html"

--------------------------------------------------------------------------------
-- Extract text of the first "# Heading" line from raw markdown.
extractH1 :: String -> String
extractH1 content =
    case filter ("# " `isPrefixOf`) (lines content) of
        (h:_) -> drop 2 h
        []    -> "Untitled"

-- Remove the first "# Heading" line to avoid duplicate title in rendered page.
-- (default.html already renders $title$ as <h1> above $body$.)
stripH1 :: String -> String
stripH1 content =
    let ls = lines content
    in unlines $ case break ("# " `isPrefixOf`) ls of
        (before, _:after) -> before ++ after
        (before, [])      -> before

-- Escape a string for safe use as a YAML double-quoted value.
escapeForYaml :: String -> String
escapeForYaml s = "\"" ++ concatMap esc s ++ "\""
  where
    esc '"'  = "\\\""
    esc '\\' = "\\\\"
    esc c    = [c]

--------------------------------------------------------------------------------
-- Convert Obsidian wikilinks [[target]] to standard markdown links.
processWikilinks :: String -> String
processWikilinks []               = []
processWikilinks ('[':'[':rest)   =
    case break (== ']') rest of
        (target, ']':']':after) ->
            "[" ++ target ++ "](/notes/" ++ slugify target ++ "/)"
            ++ processWikilinks after
        _ -> '[' : '[' : processWikilinks rest
processWikilinks (c:rest)         = c : processWikilinks rest

slugify :: String -> String
slugify = map (\c -> if c == ' ' then '-' else toLower c)

--------------------------------------------------------------------------------
-- Pandoc AST transform: convert Obsidian callout blockquotes to styled divs.
-- > [!NOTE] body  →  <div class="callout callout-note">body</div>
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

--------------------------------------------------------------------------------
-- Writer options for notes (same as pandocCompiler_ but reused here).
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
           { writerExtensions     = newExts
           , writerHTMLMathMethod = MathJax ""
           }

--------------------------------------------------------------------------------
-- Compiler for individual notes:
--   1. Extract H1 title, inject as YAML frontmatter (so $title$ works in templates)
--   2. Strip H1 from body (default.html already renders $title$ as <h1>)
--   3. Convert wikilinks to standard markdown links
--   4. Render with Pandoc (obsidianTransform will be wired in when we
--      upgrade to Hakyll >= 4.13 which exports renderPandocWithTransform)
noteCompiler :: Compiler (Item String)
noteCompiler = do
    raw <- getResourceString
    let rawBody   = itemBody raw
        title     = extractH1 rawBody
        withMeta  = "---\ntitle: " ++ escapeForYaml title ++ "\n---\n"
                    ++ stripH1 rawBody
        processed = processWikilinks withMeta
    makeItem processed
        >>= renderPandocWith defaultHakyllReaderOptions noteWriterOptions

