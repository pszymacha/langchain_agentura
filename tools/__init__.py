"""
Pakiet zawierający implementacje narzędzi dla agenta.
"""

from typing import Dict, Type
from .base_tool import AgentTool
from .duckduckgo_tool import DuckDuckGoTool
from .wikipedia_tool import WikipediaTool
from .math_tool import MathTool
from .python_tool import PythonTool
from .datetime_tool import DateTimeTool

# Rejestr wszystkich dostępnych narzędzi
TOOL_REGISTRY: Dict[str, Type[AgentTool]] = {
    "duckduckgo": DuckDuckGoTool,
    "wikipedia": WikipediaTool,
    "math": MathTool,
    "python": PythonTool,
    "datetime": DateTimeTool
} 