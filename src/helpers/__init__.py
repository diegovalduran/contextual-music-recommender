"""
Dynamic module exporter for the helpers package.
This __init__.py automatically discovers and exports all Python modules in the helpers directory 
except for this file itself.

Note: This code is adapted from the original SiTunes code.
"""

from os.path import dirname, basename, isfile, join
import glob

# Find all Python files in the current directory
modules = glob.glob(join(dirname(__file__), "*.py"))

# Create a list of module names by getting the base filename, removing the .py 
# extension, and excluding __init__.py
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')
]
