"""Single source of truth for the InfEngine product version.

Kept dependency-free so the unified CLI can print ``infengine version`` without
importing torch or the engine stack.
"""

__version__ = "0.1.0"

# The control-contract / desired-state schema generation. Bumped only when a
# persisted-artifact format changes incompatibly.
CONTROL_CONTRACT = "v2"
PRODUCT = "InfEngine"
