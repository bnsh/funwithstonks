PYTHON=$(sort $(wildcard *.py))
PYLINT3=$(addprefix .,$(PYTHON:py=pylint3))

all: pylint

pylint: $(PYLINT3)

.%.pylint3: %.py
	python3 -m pylint -r n $(^)
	@>$(@)
