# -*- coding: utf-8 -*-
"""Produit script_soutenance.tex (texte oral 45 min + questions-réponses du jury)."""
from script_data import SLIDES, QA

def esc(s):
    rep = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
           '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\^{}'}
    return ''.join(rep.get(c, c) for c in s)

def mmss(t): return f"{t//60:02d}:{t%60:02d}"

out = [r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.1cm]{geometry}
\usepackage{fontspec}\usepackage{babel}\babelprovide[import,main]{french}\babelfont{rm}{Latin Modern Roman}
\emergencystretch=3em
\usepackage{newunicodechar}
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{τ}{\ensuremath{\tau}}
\newunicodechar{η}{\ensuremath{\eta}}
\newunicodechar{²}{\ensuremath{^2}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{↔}{\ensuremath{\leftrightarrow}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{±}{\ensuremath{\pm}}
\usepackage{xcolor,titlesec,fancyhdr,enumitem,tabularx,booktabs}
\usepackage[hidelinks]{hyperref}
\definecolor{navy}{HTML}{16324F}\definecolor{gris}{HTML}{5A6470}\definecolor{rouge}{HTML}{A32020}
\titleformat{\section}{\Large\bfseries\color{navy}}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries\color{navy}}{}{0pt}{}
\titlespacing*{\subsection}{0pt}{10pt}{3pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\color{gris}Soutenance -- N.~Belhadj}\fancyhead[R]{\small\color{gris}Document de préparation}
\fancyfoot[C]{\small\thepage}
\newcommand{\slidehead}[4]{\subsection*{\colorbox{navy}{\color{white}\strut\,#1\,}\quad #2\hfill{\small\color{gris}\mdseries #3 \textbar{} durée #4}}}
\begin{document}
\begin{center}
{\LARGE\bfseries\color{navy} Texte de la soutenance et questions du jury}\\[4pt]
{\large Deep Learning-Based Prediction of Disease Severity Using Medical Imaging:\\ An Application to Diabetology}\\[4pt]
{\small Nader Belhadj -- durée de l'exposé : 45 minutes -- 39 diapositives (\texttt{Belhadj\_PhD\_Defence.pptx})}
\end{center}

\section*{Mode d'emploi}
\begin{itemize}[leftmargin=1.4em,itemsep=2pt]
\item Le texte de chaque diapositive est calibré pour un rythme posé d'environ 100 à 110 mots par minute, pauses comprises ; le total fait 45 minutes. L'horodatage indique quand commencer et terminer chaque diapositive.
\item Le même texte figure dans les notes du PowerPoint (mode Présentateur).
\item Ne lisez pas : apprenez les idées et les chiffres, et gardez les phrases de transition.
\item Si vous êtes en retard, raccourcissez d'abord les diapositives 13, 14 et 31 ; ne coupez jamais les diapositives 19 (H1), 21 (H2) et 29 (H3a), où se jouent les questions du jury.
\item Tous les chiffres viennent de la thèse (version v29).
\end{itemize}

\section*{Partie A -- Texte oral, diapositive par diapositive}
"""]
t = 0
for i, (title, d, text) in enumerate(SLIDES, 1):
    start = t; t += d
    out.append(r"\slidehead{%d}{%s}{%s -- %s}{%s}" % (i, esc(title), mmss(start), mmss(t), f"{d//60}:{d%60:02d}"))
    out.append(esc(text) + "\n")
out.append(r"""
\medskip{\color{gris}\small Total : %s.}

\newpage
\section*{Partie B -- Questions probables du jury et réponses}
Principe : répondre en deux ou trois phrases, commencer par la réponse directe, citer un chiffre, reconnaître la limite quand elle existe, puis s'arrêter. Il vaut mieux dire « c'est une limite, et voici comment je la traiterais » que défendre un point faible.
""" % mmss(t))
theme = None; k = 0
for th, q, a in QA:
    if th != theme:
        out.append(r"\subsection*{%s}" % esc(th)); theme = th
    k += 1
    out.append(r"\textbf{Q%d.~%s}\par\nopagebreak" % (k, esc(q)))
    out.append(r"\textit{Réponse.}~" + esc(a) + r"\par\medskip")
out.append(r"""
\section*{Partie C -- Formulations à employer et à éviter}
\begin{tabularx}{\linewidth}{@{}>{\color{rouge}}X>{\raggedright\arraybackslash}X@{}}
\toprule
À éviter & À dire \\
\midrule
« Nous avons prouvé que le graphe améliore le classement » & « Les expériences apportent des preuves que l'information relationnelle améliore le classement, dans les conditions évaluées » \\
« La topologie explique la sévérité » & « Une association statistique faible entre un descripteur topologique et la sévérité » \\
« DOTS est sûr » & « Ses performances observées satisfont les objectifs de dépistage en estimation ponctuelle » \\
« Validé cliniquement » & « Évalué rétrospectivement sur des données publiques externes » \\
« Conforme à la réglementation » & « Conçu en tenant compte des exigences réglementaires » \\
« Patients » (pour les effectifs) & « Images rétiniennes » \\
« Topologie de la rétine » (pour DGTS) & « Régularisation topologique dans l'espace des caractéristiques » \\
\bottomrule
\end{tabularx}
\end{document}
""")
open('script_soutenance.tex', 'w').write('\n'.join(out))
print('ok', mmss(t))
