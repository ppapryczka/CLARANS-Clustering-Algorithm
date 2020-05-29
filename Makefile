test:
	python -mpytest -q src/ tests/

test_full:
	python -mpytest src/ tests/

cov:
	python -mpytest --cov-config=setup.cfg --cov-report html --cov=src src/ tests/
