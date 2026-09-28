Nader Belhadj -- PhD thesis (ENSI, University of Manouba) -- Overleaf project
==========================================================================

Compile
-------
Overleaf: New Project -> Upload Project -> this ZIP. Main file: main.tex.
Compiler: pdfLaTeX (default) or XeLaTeX / LuaLaTeX -- the preamble detects
the engine. BibTeX runs automatically (style: unsrt, file references.bib).

Structure
---------
main.tex                 document order (front matter, chapters, bibliography,
                         publications, appendices)
preamble.tex             layout, fonts, headings, captions, colours, links
chapters/title.tex       cover page -- see "Before printing" below
chapters/*.tex           intro, ch1 (DR and automated grading), ch2_background
                         (graphs, topology, ordinal prediction), ch2 (first
                         contributions), ch3 (TOPORET), ch4 (DOTS), ch6 (software,
                         regulation, roadmap), conclusion, publications,
                         appendix (A results, B hyperparameters, C statistics,
                         D reproducibility)
figures/*.tex            all diagrams in TikZ / pgfplots (vector, editable)
img/                     photographs only (fundus, CLAHE, skeleton, logos)

Before printing
---------------
1. Cover page: in chapters/title.tex, replace the blank lines of the jury
   table and \defencedate with the real names and date.
2. Recto-verso printing: in main.tex replace "oneside,openany" by
   "twoside,openright" (chapters will then start on odd pages; blank
   versos are left empty automatically).

Conventions
-----------
- Figure captions below, table captions above; notes under a table use
  \tabnote{...} and never repeat "Table X".
- All cross-references use \cref{...}; never type a number by hand.
- Short captions in [ ] feed the List of Figures / List of Tables.
- Front matter and General Introduction are numbered 1, 2, ...;
  the General Conclusion uses GC.1, GC.2, ...
