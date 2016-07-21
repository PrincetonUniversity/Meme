REPORT=paper
TARGET=paper
CAMERA=sigc056
LATEX=pdflatex
BIBTEX=bibtex --min-crossrefs=100
REF1=ref
REF2=rfc

TEX = $(wildcard *.tex)
SRCS = abstract.tex introduction3.tex hardtoday.tex sdx-abstractions.tex \
	compilation.tex compiler-optimizations.tex implementation.tex evaluation.tex related.tex conclusion.tex
#SRCS = $(TEX)
REFS=$(REF1).bib $(REF2).bib

all: pdf

############################################################

camera: camera-clean $(CAMERA).ps $(CAMERA).pdf $(CAMERA).tex 
camera-clean: 
	rm -f $(CAMERA).*

$(CAMERA).tex: 
	rm -f $@
	cat *.tex *.bbl > $@
	perl -pi -e 's/^\s*%.*\n//' $@
	sed -i -e 's/[^\]%.*//' $@
	sed -i -e '/^\\if 0[ ^t]*$$/,/^\\fi[ ^t]*$$/ d' $@
	tar cvf - $(CAMERA).* | gzip -c > $(CAMERA).tar.gz


$(CAMERA).ps: ps
	cp $(TARGET).ps $(CAMERA).ps

$(CAMERA).pdf: pdf
	cp $(TARGET).pdf $(CAMERA).pdf

############################################################

pdf:
	GS_OPTIONS=-dPDFSETTINGS=/prepress rubber -f --pdf -Wrefs -Wmisc $(TARGET)

ps: pdf
	GS_OPTIONS=-dPDFSETTINGS=/prepress pdftops -level1 $(TARGET).pdf


pdflatex:
	$(LATEX) $(TARGET)
	$(LATEX) $(TARGET)
	$(BIBTEX) $(TARGET)
	$(LATEX) $(TARGET)


rubber: $(SRCS) $(REFS)
	rubber --ps -d -Wrefs -Wmisc $(REPORT)

spell:
	for i in $(SRCS); do ispell $$i; done

double:
	for i in $(SRCS); do double.pl < $$i; done



clean:
	rm -f *~ *.cb* *.dvi *.aux *.log *.blg *.bbl *.out *.brf $(REPORT).ps
