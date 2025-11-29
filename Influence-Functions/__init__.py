# Influence-Functions package
import os
import sys

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import commonly used modules
try:
    from . import utils
    from . import influence
    from . import influence_pile
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

