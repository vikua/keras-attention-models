FLAGS=


black:
	black keras-attention-models tests setup.py

flake:
	flake8 keras-attention-models tests setup.py


test: flake
	py.test -s -v $(FLAGS) ./tests/


mypy:
	mypy keras-attention-models --ignore-missing-imports --disallow-untyped-calls --no-site-packages --strict


cov cover coverage: flake
	py.test -s -v --cov-report term --cov-report html --cov keras-attention-models ./tests
	@echo "open file://`pwd`/htmlcov/index.html"


cov_only: flake
	py.test -s -v --cov-report term --cov-report html --cov keras-attention-models ./tests
	@echo "open file://`pwd`/htmlcov/index.html"


install:
	pip install -r requirements-dev.txt


clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -f `find . -type f -name '@*' `
	rm -f `find . -type f -name '#*#' `
	rm -f `find . -type f -name '*.orig' `
	rm -f `find . -type f -name '*.rej' `
	rm -f .coverage
	rm -rf coverage
	rm -rf build
	rm -rf htmlcov
	rm -rf dist


doc:
	make -C docs html
	@echo "open file://`pwd`/docs/_build/html/index.html"


.PHONY: all flake test cov clean doc install
