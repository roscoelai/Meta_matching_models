#!
#
# Makefile
# 2022-06-21

SHELL := /bin/sh
PYTHON := $(shell command -v python3 2> /dev/null)
ifndef PYTHON
    $(error "Unable to find python3. Exiting.")
endif

.PHONY: usage
usage:
	@printf "Please select one of the available targets:\n\
  \033[92mcheck\033[0m    run unit tests\n\
  \033[92mclean\033[0m    remove temporary files/directories\n\
"

.PHONY: check
check:
	$(PYTHON) -m unittest tests.test_CBIG_model_pytorch

.PHONY: clean
clean:
	rm -rfv __pycache__/
	rm -rfv tests/__pycache__/

