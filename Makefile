CXX      = g++
CXXFLAGS = -O3 -fopenmp -march=native -ffast-math -std=c++17 -I include

# ── Combined benchmark (all 5 solvers) ───────────────────────────────────────
TARGET   = kmeans_bench
SRC_MAIN = src/main.cpp

# ── Standalone baseline binary ────────────────────────────────────────────────
BASELINE_TARGET = baseline_bench
BASELINE_SRC    = src/baseline/run_baseline.cpp

# ── Standalone improved binary ────────────────────────────────────────────────
IMPROVED_TARGET = improved_bench
IMPROVED_SRC    = src/improved/run_improved.cpp

HEADERS = include/common/types.hpp \
          include/baseline/seq_baseline.hpp \
          include/baseline/naive_openmp.hpp \
          include/improved/pruned_static.hpp \
          include/improved/pruned_guided.hpp \
          include/improved/proposed_solver.hpp

.PHONY: all baseline improved run run_baseline run_improved plot clean

# Build everything
all: $(TARGET) $(BASELINE_TARGET) $(IMPROVED_TARGET)
	@echo "All binaries built."

# Combined benchmark
$(TARGET): $(SRC_MAIN) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRC_MAIN) -o $(TARGET)
	@echo "Built: ./$(TARGET)"

# Standalone baseline
$(BASELINE_TARGET): $(BASELINE_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(BASELINE_SRC) -o $(BASELINE_TARGET)
	@echo "Built: ./$(BASELINE_TARGET)"

# Standalone improved
$(IMPROVED_TARGET): $(IMPROVED_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(IMPROVED_SRC) -o $(IMPROVED_TARGET)
	@echo "Built: ./$(IMPROVED_TARGET)"

# Run targets
run: $(TARGET)
	./$(TARGET)

run_baseline: $(BASELINE_TARGET)
	./$(BASELINE_TARGET)

run_improved: $(IMPROVED_TARGET)
	./$(IMPROVED_TARGET)

# Generate all plots
plot:
	python plot_results.py

clean:
	rm -f $(TARGET) $(BASELINE_TARGET) $(IMPROVED_TARGET)
	rm -f *.csv
	rm -f plots/baseline/*.png plots/improved/*.png plots/comparison/*.png
