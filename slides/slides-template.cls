%%%%%%%%%%%%%%%%%%
% Class Preamble
%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{slides-template}

\LoadClass[noamsthm, lualatex, aspectratio=43]{beamer}

\author{}
\institute{École Normale Supérieure}

\RequirePackage{xcolor}
\definecolor{vulm}{HTML}{7d1dd3}
\definecolor{yulm}{HTML}{ffe500}

%%%%%%%%%%
% Coloring
%%%%%%%%%%
\def\setmaincolor#1{\colorlet{main-below}{#1}}
\def\setaccents#1{\colorlet{main-accents}{#1}}
\def\setbackground#1{\colorlet{main-background}{#1}}
\def\settextcol#1{\colorlet{main-text}{#1}}

\setmaincolor{vulm!70!black}
\setaccents{yulm}
\setbackground{white}
\settextcol{black}

%%%%%%%%%%%%%%%%%%
% Required Imports
%%%%%%%%%%%%%%%%%%
\RequirePackage{cmap}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{babel}
\RequirePackage{lmodern}
\RequirePackage{texnames}
\RequirePackage{ragged2e}
\RequirePackage{setspace}
\setstretch{1.2}
\RequirePackage{xfp}

\RequirePackage{array}
\RequirePackage{booktabs}
\RequirePackage{textcomp}
\RequirePackage{multirow}
\RequirePackage{multicol}
\RequirePackage{nicematrix}
\RequirePackage{makecell}
\RequirePackage{colortbl}
\RequirePackage[backend=biber, style=alphabetic]{biblatex}
\RequirePackage{appendixnumberbeamer}
\RequirePackage{graphicx}

\RequirePackage{pdfpages}

\RequirePackage{tikz}
\RequirePackage{tikz-3dplot}
\RequirePackage{pgfplots}
\RequirePackage{pgfplotstable}
\pgfplotsset{compat=1.18}
\usetikzlibrary{decorations.pathreplacing,calligraphy}
\usetikzlibrary{automata, arrows, calc, matrix, positioning, math}
\tikzset{>=stealth}

\RequirePackage{wrapfig}
\RequirePackage{graphicx}
\RequirePackage{hyperref}
\hypersetup{
	linkbordercolor=vulm,
	citebordercolor=vulm,
	menubordercolor=yulm
}
\RequirePackage[justification=centering]{caption}

\RequirePackage{etoolbox}

\def\vsp{\vphantom{pb}}

%%%%%%%%%%%%%%%%%%
% Beamer Setup
%%%%%%%%%%%%%%%%%%

\usefonttheme[onlymath]{serif}

\mode<presentation>
\useinnertheme{rectangles}
% \useoutertheme[hooks]{tree}
\def\isspecialpage{}

\def\addsectionplans{\AtBeginSection[]{\begin{frame}{Plan}\tableofcontents[subsectionstyle=shaded/hide,sectionstyle=show/shaded]\end{frame}}}
\def\addsubsectionplans{\AtBeginSubsection[]{\begin{frame}{Plan}\tableofcontents[subsectionstyle=show/shaded/hide,sectionstyle=show/shaded]\end{frame}}}

\setbeamercolor{structure}{fg=main-below}
\setbeamercolor{frametitle}{fg=main-text}
\setbeamercolor{headline}{fg=main-accents, bg=main-below}
\setbeamercolor{footline}{fg=main-text, bg=main-background}
\setbeamercolor{footline-pbar}{bg=main-below}
\setbeamercolor{background canvas}{bg=main-background}
\setbeamertemplate{navigation symbols}{}

\newdimen\footlineheight
\newdimen\footlinedepth
\newdimen\headlineheight
\newdimen\headlinedepth
\footlineheight=3ex%
\footlinedepth=1ex%
\headlineheight=3ex%
\headlinedepth=1ex%

\newdimen\logoht
\logoht=6ex
\advance\logoht by -1em

\newdimen\logowd
\newdimen\logosp

\ExplSyntaxOn
\seq_new:N \g_logos_seq

\NewDocumentCommand{\addlogo}{m}
{
	\seq_put_right:Nn \g_logos_seq { #1 }
}

\NewDocumentCommand{\clearlogos}{}
{
	\seq_clear:N \g_logos_seq
}

\NewDocumentCommand{\headlinelogos}{}
{
	\seq_map_inline:Nn  \g_logos_seq
	{
		\hskip12pt \includegraphics[height=\logoht]{ ##1 }
	}
}

\NewDocumentCommand{\nlogos}{}{\seq_count:N{\g_logos_seq}}

\NewDocumentCommand{\titlepagelogos}{}
{
	\pgfmathtruncatemacro{\nlogos}{\seq_count:N{\g_logos_seq}}
	\ifnum\nlogos>0
		\pgfmathsetmacro{\logolength}{.4 / \nlogos}
		\pgfmathsetmacro{\skiplength}{.05 / \nlogos}
		\logowd=\logolength\textwidth
		\logosp=\skiplength\textwidth
		\seq_map_inline:Nn  \g_logos_seq
		{
			\hskip\logosp \includegraphics[width=\logowd]{ ##1 } \hskip\logosp
		}
	\fi
}

\NewExpandableDocumentCommand{\nRoman}{m}
{
	\int_to_Roman:n { #1 }
}

\ExplSyntaxOff

\makeatletter

\setbeamertemplate{headline}{%
	\ifx\isspecialpage\@empty
		\begin{beamercolorbox}[wd=\paperwidth,ht=\headlineheight,dp=\headlinedepth,sep=0pt]{headline}%
			\begin{tikzpicture}[remember picture, overlay]
				% Get page dimensions
				\coordinate (topleft) at (current page.north west);
				\coordinate (topright) at (current page.north east);
				%
				\fill[headline.bg!80] (topleft) rectangle ($(topright) + (0, -9ex)$);
				\fill[headline.bg!90] (topleft) rectangle ($(topright) + (0, -6ex)$);
				\fill[headline.bg] (topleft) rectangle ($(topright) + (0, -3ex)$);
				\fill[gray, opacity=.2] (topleft) rectangle ($(topright) + (0, -9ex)$);
				%
				\draw[black] ($(topleft) + (0, -9ex)$) -- ($(topright) + (0, -9ex)$);

				%
				\node[headline.fg, anchor=north west] (hd-title) at ($(topleft) + (1em, 0)$) {\vsp\insertshorttitle};
				\node[headline.fg, anchor=north west] (hd-sec) at ($(hd-title.north west) + (3ex, -3ex)$) {\vsp\insertsectionhead};
				\node[headline.fg, anchor=north west] (hd-subsec) at ($(hd-sec.north west) + (3ex, -3ex)$) {\vsp\insertsubsectionhead};
				%
				\ifx\headlogos\@empty\relax
				\else
					\node[anchor=north east, black] at (topright) {\headlinelogos};
				\fi
				%
				\ifx\insertsectionhead\@empty\relax
				\else
					\draw[headline.fg] ($(hd-title.west) + (0, 0)$) |- (hd-sec.west);
					\ifx\insertsubsectionhead\@empty\relax
					\else
						\draw[headline.fg] ($(hd-sec.west) + (0, 0)$) |- (hd-subsec.west);
					\fi
				\fi
			\end{tikzpicture}%
		\end{beamercolorbox}
		\vspace{0.3cm} % Add space below header
	\else
	\fi
}

\defbeamertemplate{headline}{appendix}{
	\ifx\isspecialpage\@empty
		\begin{beamercolorbox}[wd=\paperwidth,ht=\headlineheight,dp=\headlinedepth,sep=0pt]{headline}%
			\begin{tikzpicture}[remember picture, overlay]
				% Get page dimensions
				\coordinate (topleft) at (current page.north west);
				\coordinate (topright) at (current page.north east);
				%
				\fill[headline.bg!80] (topleft) rectangle ($(topright) + (0, -9ex)$);
				\fill[headline.bg!90] (topleft) rectangle ($(topright) + (0, -6ex)$);
				\fill[headline.bg] (topleft) rectangle ($(topright) + (0, -3ex)$);
				\fill[gray, opacity=.2] (topleft) rectangle ($(topright) + (0, -9ex)$);
				%
				\draw[black] ($(topleft) + (0, -9ex)$) -- ($(topright) + (0, -9ex)$);

				%
				\node[headline.fg, anchor=north west] (hd-title) at ($(topleft) + (1em, -0ex)$) {\vsp\insertshorttitle};
				\node[headline.fg, anchor=north west] (hd-sec) at ($(hd-title.north west) + (3ex, -3ex)$) {\vsp Appendices};
				%
				\node[anchor=north east, black] at (topright) {\headlinelogos};
				%
				\draw[headline.fg] ($(hd-title.west) + (0, 0)$) |- (hd-sec.west);
			\end{tikzpicture}%
		\end{beamercolorbox}
		\vspace{0.3cm} % Add space below header
	\else
	\fi

}

\newdimen\progressb@r

\setbeamertemplate{footline}{%
	\leavevmode%
	\ifx\isspecialpage\@empty
		%
		\begin{beamercolorbox}[wd=\paperwidth,ht=\footlineheight,dp=\footlinedepth,sep=0pt]{footline}%
			% Text content first (background layer)
			\hbox to \paperwidth{%
				\begin{beamercolorbox}[wd=.33\paperwidth,ht=\footlineheight,dp=\footlinedepth,left,leftskip=1em]{footline}%
					\usebeamerfont{footline}\insertshortauthor\vsp%
				\end{beamercolorbox}%
				\begin{beamercolorbox}[wd=.34\paperwidth,ht=\footlineheight,dp=\footlinedepth,center]{footline}%
					\usebeamerfont{footline}\insertinstitute\vsp%
				\end{beamercolorbox}%
				\begin{beamercolorbox}[wd=.33\paperwidth,ht=\footlineheight,dp=\footlinedepth,right,rightskip=1em]{footline}%
					\usebeamerfont{footline}%
					\insertshortdate\quad\insertframenumber\,/\,\insertmainframenumber\vsp%
				\end{beamercolorbox}%
			}%
			%
			% Progress bar using TikZ (overlay on top)
			\begin{tikzpicture}[overlay,remember picture]
				\coordinate (footstart) at (current page.south west);
				\coordinate (footend) at (current page.south east);
				%
				% Background bar
				\fill[structure.fg!20]
				([yshift=.75\footlineheight]footstart) rectangle ([yshift=\footlineheight]footend);
				%
				% Progress bar
				\ifx\progbar\@empty
				\else
					\pgfmathsetmacro{\progb@rsteps}{1/\insertmainframenumber}
					\pgfmathsetmacro{\isdegenerate}{\progb@rsteps > 0.6 ? 1 : 0}
					\ifnum\isdegenerate=0
						\global\progressb@r=\progb@rsteps\paperwidth
					\else
						\typeout{Step = 1, progress set to 0 instead}
						\global\progressb@r=0pt
					\fi
					\fill[structure.fg!80]
					([yshift=.75\footlineheight]footstart) rectangle
					([yshift=\footlineheight,xshift=\insertframenumber\progressb@r]footstart);
				\fi
			\end{tikzpicture}%
		\end{beamercolorbox}%
		\vskip0pt%
	\else
	\fi
}

\defbeamertemplate{footline}{appendix}
{
	\leavevmode%
	\ifx\isspecialpage\@empty
		\begin{beamercolorbox}[wd=\paperwidth,ht=\footlineheight,dp=\footlinedepth,sep=0pt]{footline}%
			% Text content first (background layer)
			\hbox to \paperwidth{%
				\begin{beamercolorbox}[wd=.33\paperwidth,ht=\footlineheight,dp=\footlinedepth,left,leftskip=1em]{footline}%
					\usebeamerfont{footline}\insertshortauthor\vsp%
				\end{beamercolorbox}%
				\begin{beamercolorbox}[wd=.34\paperwidth,ht=\footlineheight,dp=\footlinedepth,center]{footline}%
					\usebeamerfont{footline}\insertinstitute\vsp%
				\end{beamercolorbox}%
				\begin{beamercolorbox}[wd=.33\paperwidth,ht=\footlineheight,dp=\footlinedepth,right,rightskip=1em]{footline}%
					\usebeamerfont{footline}%
					\insertshortdate\quad\Roman{framenumber}\,/\,\nRoman{\appendixtotalframenumber}\vsp%
				\end{beamercolorbox}%
			}%
			\begin{tikzpicture}[overlay,remember picture]
				\coordinate (footstart) at (current page.south west);
				\coordinate (footend) at (current page.south east);
				%
				% Background bar
				\fill[structure.fg] ([yshift=.75\footlineheight]footstart) rectangle ([yshift=\footlineheight]footend);
			\end{tikzpicture}%

		\end{beamercolorbox}%
		\vskip0pt%
	\else
	\fi
}

\g@addto@macro\appendix{%
	\setbeamertemplate{footline}[appendix]%
	\setbeamertemplate{headline}[appendix]%
}

\newdimen\frametitleleftmargin
\frametitleleftmargin=.666666\beamer@leftmargin
\advance\frametitleleftmargin by -.333333em

\newdimen\frametitlerightmargin
\frametitlerightmargin=\frametitleleftmargin
\advance\frametitlerightmargin by \textwidth

\setbeamertemplate{frametitle}{
	\begin{beamercolorbox}[wd=\paperwidth,ht=1.2cm,dp=0.3cm]{frametitle}
		\begin{tikzpicture}
			\node (tmp) at (0, 0) {};
			\node[anchor=west, inner xsep=0pt] (title) at (\frametitleleftmargin, 0cm) {\textbf{\color{frametitle.fg}\Large\insertframetitle\vsp}};
			\ifx\insertframesubtitle\@empty
				\draw[structure.fg] let \p1 = (title.south west) in (title.south west) -- ($(title.south west)!.6666!(\frametitlerightmargin, \y1)$);
			\else
				\node[anchor=west] (subtitle) at ($(title) + (.5cm, -\baselineskip)$) {\small\insertframesubtitle};
				\draw[frametitle.bg] ($(title.south west) + (.333333em, 0)$) |- (subtitle.west);
			\fi
		\end{tikzpicture}
	\end{beamercolorbox}
	\vspace{-.3cm}
}

\makeatother

\RequirePackage{adforn}

\newsavebox{\tileone}

\def\textbgopac{.6}
\newcommand{\fancytitleframe}{
	\def\isspecialpage{1}
	\begin{frame}
		\sbox{\tileone}{%
			\begin{tikzpicture}[x=1cm,y=1cm]
				\clip (0,0) rectangle (2,2);
				\foreach \x in {0,1} {
						\foreach \y in {0,1} {
								\node[font=\Large,text=main-below!20]
								at (2*\x, 2*\y) {$\beth$};
								\node[font=\Large,text=main-accents!80!black!40]
								at (2*\x+1, 2*\y+1) {$\aleph$};
							}
					}
				\begin{scope}[dashed, double, main-below!50]
					\draw plot (\x,\x-1);
					\draw plot (\x,\x+1);
					\draw plot (\x,-\x+1);
					\draw plot (\x,-\x+3);
				\end{scope}
			\end{tikzpicture}%
		}
		\begin{tikzpicture}[overlay,remember picture]
		\end{tikzpicture}
		\begin{tikzpicture}[remember picture, overlay]
			\node (upperleft) at (current page.north west) {};
			\path[fill opacity=.5, fill tile picture={%
						\node[inner sep=0pt,outer sep=0pt] {\usebox{\tileone}};
					}] (upperleft) rectangle (current page.south east);
			%
			\node [anchor=north, rectangle, fill=main-below, fill opacity=.8, text opacity=1, minimum width=\paperwidth, align=center, yshift=-3\headlineheight, inner ysep=2em] (title) at (current page.north)
			{
			{\parbox{.9\paperwidth}{\centering\textcolor{main-background}{\Large\bf\inserttitle}}}
			\\[.5\headlineheight]
			{\parbox{.9\paperwidth}{\centering\textcolor{main-background}{\bf\insertsubtitle}}}
			};
			%
			\node [anchor=north, rectangle, fill=main-background, fill opacity=\textbgopac, text opacity=1, yshift=-3ex] (author) at (title.south) {\insertauthor};
			%
			\node [anchor=north, rectangle, fill=main-background, fill opacity=\textbgopac, text opacity=1, yshift=-3ex] (logos) at (author.south) {\titlepagelogos};
			\node [anchor=north, rectangle, fill=main-background, fill opacity=\textbgopac, text opacity=1, yshift=-3ex] (date) at (logos.south) {\insertdate};
			%
			\node [anchor=south, yshift=\footlineheight, minimum width=\paperwidth, text width=0.9\paperwidth, rectangle, fill=main-background, fill opacity=\textbgopac, text opacity=1, align=center] (institutes) at (current page.south) {\insertinstitute};
		\end{tikzpicture}
	\end{frame}
	\def\isspecialpage{}
}

\newcommand{\questionsframe}{%
	\def\isspecialpage{1}
	\begin{frame}
		\sbox{\tileone}{%
			\begin{tikzpicture}[x=1cm,y=1cm]
				\clip (0,0) rectangle (2,2);
				\foreach \x in {0,1} {
						\foreach \y in {0,1} {
								\node[font=\Large,text=main-below!20]
								at (2*\x, 2*\y) {$\beth$};
								\node[font=\Large,text=main-accents!80!black!40]
								at (2*\x+1, 2*\y+1) {$\aleph$};
							}
					}
				\begin{scope}[dashed, double, main-below!50]
					\draw plot (\x,\x-1);
					\draw plot (\x,\x+1);
					\draw plot (\x,-\x+1);
					\draw plot (\x,-\x+3);
				\end{scope}
			\end{tikzpicture}%
		}
		\begin{tikzpicture}[overlay, remember picture]
		\end{tikzpicture}
		\begin{tikzpicture}[remember picture, overlay]
			\node (upperleft) at (current page.north west) {};
			\path[fill opacity=.5, fill tile picture={%
						\node[inner sep=0pt,outer sep=0pt] {\usebox{\tileone}};
					}] (upperleft) rectangle (current page.south east);
			%
			\node[fill=main-background, text opacity=1, fill opacity=.8, rectangle, draw=none, minimum width=.8\paperwidth, align=center, inner ysep=2em] (text) at (current page.center) {%
				\langopt{Thank you for your attention.\\[2em]
					Do you have any questions?}%
				{Merci de votre attention.\\[2em]
					Avez-vous des questions ?}%
			};
			% \draw[main-below] (text.south west) -| (text.north east);
		\end{tikzpicture}
	\end{frame}
	\def\isspecialpage{}
}


%%%%%%%%%%%%%%%%%%
% Document Macros
%%%%%%%%%%%%%%%%%%
\newcommand{\tocless}[2]{\bgroup\let\addcontentsline=\nocontentsline#1{#2}\egroup}
