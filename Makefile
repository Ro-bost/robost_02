PYTHON ?= python
export MUJOCO_GL ?= egl

.PHONY: check test test-sim test-rl build-swing

check:
	$(PYTHON) -m compileall -q src tests scripts

test: test-sim

test-sim:
	$(PYTHON) -m unittest tests.test_validation -v
	$(PYTHON) -m unittest tests.test_stairs_cli -v
	$(PYTHON) -m unittest tests.test_terrain -v
	$(PYTHON) -m unittest tests.test_walk -v

test-rl:
	$(PYTHON) -m unittest discover -s tests -p 'test_rl_*.py' -v

build-swing:
	rs02-build-swing
