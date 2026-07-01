"""Shared pytest fixtures/config for the GlyphNet test suite.

Adds the `code/` directory to sys.path so the pipeline scripts can be imported
by their module names (dataGeneration, train, attentionModule, ...).
"""
import os
import sys

CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
