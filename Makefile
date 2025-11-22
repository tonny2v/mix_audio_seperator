.PHONY: help run-timeline-example

help:
	@echo "Available targets:"
	@echo "  run-timeline-example - Run timeline separation on mix.flac example"
	@echo "  help                - Show this help message"

run-timeline-example:
	python timeline_separate.py --auth-token $(HF_AUTH_TOKEN) --output-dir . examples/mix.flac