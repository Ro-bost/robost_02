PYTHON ?= python
export MUJOCO_GL ?= egl

.PHONY: check lint format test test-core test-rl

check:
	$(PYTHON) -m compileall -q src tests scripts

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff format src tests scripts

test: test-core test-rl

test-core:
	$(PYTHON) -m unittest tests.test_model tests.test_scene tests.test_terrain tests.test_validation tests.test_stairs_cli tests.test_evaluate_cli tests.test_policy_cli tests.test_render_trial -v

test-rl:
	$(PYTHON) -m unittest discover -s tests -p 'test_rl_*.py' -v
