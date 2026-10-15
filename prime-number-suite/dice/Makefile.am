bin_PROGRAMS = dice_cpu
dice_cpu_SOURCES = src/dice_cpu.cpp
AM_CXXFLAGS = -Wall -Wextra -Wpedantic

TESTS = \
	tests/spec_parsing.sh \
	tests/determinism.sh \
	tests/range.sh \
	tests/chi_square.sh

AM_TESTS_ENVIRONMENT = \
	DICE_CPU='$(abs_top_builddir)/dice_cpu$(EXEEXT)'; export DICE_CPU;

EXTRA_DIST = $(TESTS) tests/common.sh README.md
