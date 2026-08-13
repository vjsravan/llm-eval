.PHONY: test baseline demo report clean

PY      ?= python3
SUITE   := examples/customs_classification/suite.json
export PYTHONPATH := $(CURDIR):$(CURDIR)/examples/customs_classification

test:
	$(PY) -m unittest discover -s tests

# Regenerate the committed baseline. Run this deliberately, when a change to the prompt
# or the suite is intended — never to make a red gate go green.
baseline:
	$(PY) -m llmeval.cli run $(SUITE) \
	  --model models:baseline_model \
	  --label baseline \
	  --save runs/baseline.json

# The demo: the same suite under a prompt that dropped two guardrails. Exits 1, which is
# the point — this is what the gate blocking a merge actually looks like.
demo:
	-$(PY) -m llmeval.cli run $(SUITE) \
	  --model models:regressed_model \
	  --label regressed-demo \
	  --baseline runs/baseline.json \
	  --save runs/regressed.json \
	  --html runs/report.html \
	  --require-tag safety=1.0 \
	  --require-tag hallucination-guard=1.0

report: runs/regressed.json
	$(PY) -m llmeval.cli report runs/regressed.json \
	  --baseline runs/baseline.json \
	  --html runs/report.html

clean:
	rm -f runs/regressed.json runs/candidate.json runs/report.html
