# PWARM — one-command credibility
# On Windows use Git Bash (make ships with it) or WSL.

PYTHON ?= python
export PYTHONPATH := src

.PHONY: demo demo-001 demo-002 demo-003 test repro clean

demo: demo-001            ## the stranger test: watch the AI discover gravity

demo-001:                 ## Mission 001 — gravity discovery (headless)
	$(PYTHON) -m pymo.cli demo 001

demo-002:                 ## Mission 002 — autonomous material discovery (headless)
	$(PYTHON) -m pymo.cli demo 002

demo-003:                 ## Mission 003 — budget + instrument arc (headless)
	$(PYTHON) -m pymo.cli demo 003

test:                     ## full test suite
	$(PYTHON) -m pytest tests/ -q

repro:                    ## regenerate all reproducibility results
	bash reproducibility/mission_001/run.sh
	bash reproducibility/mission_002/run.sh
	bash reproducibility/mission_003/run.sh

clean:                    ## remove generated results + knowledge
	rm -rf reproducibility/mission_*/results
	rm -f knowledge/universe_*.json
