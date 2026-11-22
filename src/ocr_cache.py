from __future__ import annotations

import threading

# Global thread-safe background OCR prefetch cache.
# Stored in this imported module so it persists across Streamlit page reruns
# (Streamlit executes page scripts in a new fake __main__ module on each run).
PREFETCH_CACHE: dict[int, dict] = {}
PREFETCH_LOCK = threading.Lock()
PREFETCH_IN_PROGRESS: set[int] = set()
