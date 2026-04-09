{-# LANGUAGE OverloadedStrings #-}

import           Control.Monad (forM_)
import           Hakyll
import           Text.Pandoc.Options

import           Data.Char (toLower, toUpper)
import           Data.List (isPrefixOf, nub)
import           Data.Maybe (mapMaybe)
import qualified Data.Text as T
import           System.FilePath (dropExtension, joinPath, splitDirectories, takeDirectory, (</>))
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

    match (fromList ["about.rst"]) $ do
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

    -- Copy non-markdown assets from notes (figures, images in concepts/papers)
    match ("notes/**" .&&. complement "notes/**.md") $ do
        route   idRoute
        compile copyFileCompiler

    -- Render individual note pages
    -- Pattern covers concepts, papers, curricula but excludes CLAUDE.md, README,
    -- docs/, plans/, and .claude/ by only matching the three known content sections.
    let notePattern =
          ("notes/concepts/**.md" .||. "notes/papers/**.md" .||. "notes/curricula/**.md")
          .&&. complement ("notes/**/exercises.md" .||. "notes/**/solutions.md")
    match notePattern $ do
        route $ customRoute noteRoute
        compile $ do
            -- Extract title from H1 before compilation so it's available in
            -- the template context (Hakyll's metadata store only reads the
            -- original source file, which has no YAML frontmatter).
            raw <- getResourceString
            let title = extractH1 (itemBody raw)
            -- Save title string as a snapshot so directory index pages can load it.
            titleItem <- makeItem title
            _ <- saveSnapshot "note-title" titleItem
            let noteCtx = constField "title" title `mappend` defaultContext
            noteCompiler
                >>= loadAndApplyTemplate "templates/note.html"    noteCtx
                >>= loadAndApplyTemplate "templates/default.html" noteCtx
                >>= relativizeUrls

    -- Generate index pages for every directory in the notes tree, at any depth.
    -- This preserves the full folder hierarchy without hardcoding depth levels.
    noteIds <- getMatches notePattern
    let notePaths = map toFilePath noteIds
        -- Every ancestor directory of every note (from "notes/" down to the note's parent)
        notesDirs = nub $ concatMap (parentDirs . toFilePath) noteIds

    forM_ notesDirs $ \dir -> do
        let subdirs    = immediateSubdirs dir notePaths
            dirNoteIds = immediateNotes   dir noteIds
            dirTitle   = if dir == "notes" then "Notes" else humanize (last (splitDirectories dir))
            tmpl       = if dir == "notes"
                         then "templates/notes-index.html"
                         else "templates/notes-dir.html"
        create [fromFilePath (dir ++ "/index.html")] $ do
            route idRoute
            compile $ do
                noteItems <- loadAll (fromList dirNoteIds)
                let subdirsCtx =
                        field "name" (return . humanize . last . splitDirectories . itemBody)
                        `mappend` field "url" (\i -> return $ "/" ++ itemBody i ++ "/")
                    -- Load the title saved by noteCompiler's saveSnapshot "note-title"
                    noteItemCtx =
                        field "title" (\item ->
                            itemBody <$> loadSnapshot (itemIdentifier item) "note-title")
                        `mappend` defaultContext
                    dirCtx =
                        listField "subdirs" subdirsCtx   (mapM makeItem subdirs) `mappend`
                        listField "notes"   noteItemCtx  (return noteItems)       `mappend`
                        constField "title"  dirTitle                              `mappend`
                        defaultContext
                makeItem ""
                    >>= loadAndApplyTemplate tmpl                          dirCtx
                    >>= loadAndApplyTemplate "templates/default.html"      dirCtx
                    >>= relativizeUrls

    match "templates/**" $ compile templateBodyCompiler


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

capitalize :: String -> String
capitalize []     = []
capitalize (c:cs) = toUpper c : cs

-- "ab-testing" → "Ab Testing"
humanize :: String -> String
humanize = unwords . map capitalize . splitOn '-'
  where
    splitOn _ [] = []
    splitOn d s  = let (w, rest) = break (== d) s
                   in w : case rest of { [] -> []; (_:t) -> splitOn d t }

-- All ancestor directory paths of a file path, from root down to direct parent.
-- "notes/concepts/a/b/foo.md" → ["notes", "notes/concepts", "notes/concepts/a", "notes/concepts/a/b"]
parentDirs :: FilePath -> [FilePath]
parentDirs path =
    let parts = splitDirectories (takeDirectory path)
        n     = length parts
    in [joinPath (take k parts) | k <- [1..n]]

-- Immediate subdirectory paths of dir within a set of file paths.
-- e.g. dir="notes/concepts/a", path "notes/concepts/a/b/foo.md" → Just "notes/concepts/a/b"
immediateSubdirs :: FilePath -> [FilePath] -> [FilePath]
immediateSubdirs dir paths = nub $ mapMaybe go paths
  where
    prefix = dir ++ "/"
    go path
      | prefix `isPrefixOf` path =
          let rest   = drop (length prefix) path
              subdir = takeWhile (/= '/') rest
          in if '/' `elem` rest && not (null subdir)
             then Just (dir ++ "/" ++ subdir)
             else Nothing
      | otherwise = Nothing

-- Note identifiers whose file is directly inside dir (not in a subdirectory).
immediateNotes :: FilePath -> [Identifier] -> [Identifier]
immediateNotes dir ids = filter go ids
  where
    prefix = dir ++ "/"
    go ident =
        let path = toFilePath ident
            rest = drop (length prefix) path
        in prefix `isPrefixOf` path && '/' `notElem` rest

--------------------------------------------------------------------------------
-- Pandoc AST transform: convert Obsidian callout blockquotes to styled divs.
-- > [!NOTE] body  →  <div class="callout callout-note">body</div>
obsidianTransform :: Pandoc -> Pandoc
obsidianTransform = walk transformBlock

transformBlock :: Block -> Block
transformBlock (BlockQuote (Para inlines : rest))
    | (firstStr : remainingInlines) <- inlines
    , Just calloutType <- parseCalloutMarker firstStr =
        Div ("", ["callout", "callout-" <> T.pack calloutType], [])
            (Para remainingInlines : rest)
transformBlock b = b

parseCalloutMarker :: Inline -> Maybe String
parseCalloutMarker (Str s)
    | "[!" `T.isPrefixOf` s && "]" `T.isSuffixOf` s =
        Just (T.unpack (T.toLower (T.drop 2 (T.init s))))
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
--   4. Render with Pandoc + obsidianTransform (callouts → styled divs)
noteCompiler :: Compiler (Item String)
noteCompiler = do
    raw <- getResourceString
    let rawBody   = itemBody raw
        title     = extractH1 rawBody
        withMeta  = "---\ntitle: " ++ escapeForYaml title ++ "\n---\n"
                    ++ stripH1 rawBody
        processed = processWikilinks withMeta
    makeItem processed
        >>= renderPandocWithTransform
                defaultHakyllReaderOptions
                noteWriterOptions
                obsidianTransform

