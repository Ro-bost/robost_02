PYTHON ?= python
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)
export MUJOCO_GL ?= egl

.PHONY: check test test-core test-rl build-swing release
check:
	$(PYTHON) scripts/check_project.py

test:
	$(PYTHON) -m unittest discover -s tests -v

test-core:
	$(PYTHON) -m unittest discover -s tests -p 'test_rs02_course_validation.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_run_rs02_stairs.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_rs02_terrain.py' -v
	$(PYTHON) -m unittest discover -s tests -p 'test_rs02_walk.py' -v

test-rl:
	$(PYTHON) -m unittest discover -s tests -p 'test_rs02_rl*.py' -v

build-swing:
	$(PYTHON) -m build_cheetah_swing

release:
	$(PYTHON) scripts/export_release.py
