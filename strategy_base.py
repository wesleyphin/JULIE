from abc import ABC, abstractmethod
from typing import Dict, Optional


class Strategy(ABC):
    @abstractmethod
    def on_bar(self, df) -> Optional[Dict]:
        """Return a signal dict or None for no action."""
        raise NotImplementedError
