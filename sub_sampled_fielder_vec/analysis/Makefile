.PHONY: all phase1 phase2 phase3 summary clean help

# Default target
all: phase1 phase2 phase3 summary

# Run individual phases
phase1:
	@echo "Running Phase 1: Diagnostic Metrics Analysis..."
	python3 phase1_diagnostic_metrics.py

phase2:
	@echo "Running Phase 2: Scaling Laws Analysis..."
	python3 phase2_scaling_laws.py

phase3:
	@echo "Running Phase 3: Theoretical Interpretation..."
	python3 phase3_theoretical_connection.py

# Generate master summary
summary:
	@echo "Generating master summary..."
	python3 generate_summary.py

# Run all phases using the runner script
run-all:
	python3 run_all_phases.py

# Test utilities
test:
	@echo "Testing utilities..."
	python3 utils.py

# Clean generated outputs
clean:
	@echo "Cleaning analysis outputs..."
	rm -rf ../results/combined_grid_search_results/analysis_outputs/*
	@echo "Done. All analysis outputs removed."

# Display help
help:
	@echo "Spectral Tree Reconstruction Analysis"
	@echo "======================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make all        - Run all phases and generate summary (default)"
	@echo "  make phase1     - Run Phase 1: Diagnostic Metrics"
	@echo "  make phase2     - Run Phase 2: Scaling Laws"
	@echo "  make phase3     - Run Phase 3: Theoretical Interpretation"
	@echo "  make summary    - Generate master summary document"
	@echo "  make run-all    - Run all phases using runner script"
	@echo "  make test       - Test utilities module"
	@echo "  make clean      - Remove all generated outputs"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make           # Run everything"
	@echo "  make phase1    # Run only Phase 1"
	@echo "  make clean all # Clean and rerun everything"
