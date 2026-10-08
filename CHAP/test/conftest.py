# System modules
import sys
from unittest.mock import MagicMock

# Create a custom mock that safely handles python inspection attributes
class SafeInspectionMock(MagicMock):
    def __getattr__(self, name):
        if name == '__code__':
            return None
        return super().__getattr__(name)

# Inject the safe mock into sys.modules
sys.modules['tkinter'] = SafeInspectionMock()
sys.modules['tkinter.ttk'] = SafeInspectionMock()
sys.modules['tkinter.messagebox'] = SafeInspectionMock()

