SRC ?= skais-mapper

.DEFAULT_GOAL := build

skais_mapper/raytrace.c:
	uv run python setup.py build_c

.PHONY: extensions
extensions:
	uv run python setup.py build_ext --inplace

skais_mapper/raytrace.so: skais_mapper/raytrace.c extensions
	ln -sf $(lastword $(sort $(notdir $(wildcard skais_mapper/raytrace.*.so)))) skais_mapper/raytrace.so

build: skais_mapper/raytrace.so
	uv build


.PHONY: install
install: build
	uv sync


.PHONY: clean
clean:
	find . -type d -name ".tmp" -exec rm -r {} +
	rm -f skais_mapper/raytrace.html
	rm -rf *.egg-info build dist


README.md:
	emacsclient --alternate-editor='emacs' -e '(progn (find-file "README.org") (org-gfm-export-to-markdown))'


.PHONY: test
test:
	clear; uv run pytest -sv

.PHONY: report
report:
	uv run python setup.py build_c -a
