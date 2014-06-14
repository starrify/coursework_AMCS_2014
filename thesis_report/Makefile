# This is part of coursework for Applied Mathematics for Computer Science
# by Pengyu CHEN (pengyu[at]libstarrify.so)
# COPYLEFT, ALL WRONGS RESERVED.

TEX_SRC_PREFIX=thesis_report
TEX_SRC=${TEX_SRC_PREFIX}.tex
OUTPUT_PDF=thesis_report.pdf

.PHONY: clean tmp_font_prepare tmp_font_cleanup
.DEFAULT: thesis
.SUFFIXES:

FONT_DIR_DST=~/.fonts/tmp_fonts
FONT_DIR_SRC=$(shell realpath ../tmp_fonts)

thesis: ${OUTPUT_PDF}
	$(MAKE) tmp_font_cleanup

tmp_font_prepare:
	ln -s --force ${FONT_DIR_SRC} ${FONT_DIR_DST}

tmp_font_cleanup:
	rm -f ${FONT_DIR_DST}

${OUTPUT_PDF}: clean *.tex
	$(MAKE) tmp_font_cleanup
	$(MAKE) tmp_font_prepare
	pdflatex ${TEX_SRC} 
	#find data -name "*.tex" | sed -e "s/\.tex//g" | parallel --no-notice bibtex || true
	bibtex ${TEX_SRC_PREFIX} || true
	pdflatex ${TEX_SRC} 
	pdflatex ${TEX_SRC} 

clean:
	rm -f ${OUTPUT_PDF}
	find . \( -name "*.aux" -o -name "*.toc" -o -name "*.bbl" -o -name "*.blg" \
		-o -name "*.log" -o -name "*.out" \) | xargs rm -f
