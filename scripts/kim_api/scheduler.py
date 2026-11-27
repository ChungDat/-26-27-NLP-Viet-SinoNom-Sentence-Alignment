"""Scheduled batch OCR runner for the Kim Hán Nôm API.

Implements a resumable 2-step pipeline (``image-upload`` -> ``image-ocr``)
with SQLite checkpointing, sliding-window rate limiting across minute/hour/day/week
quotas, exponential backoff retries, and outputs in raw JSON, plain text, and
PPOCRLabel (``Label.txt`` + ``fileState.txt``) formats.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import mimetypes
import os
import signal
import sqlite3
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Ensure project root is importable when invoked as a standalone script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ppocr import (  # noqa: E402
    PPOCRLabelAnnotation,
    PPOCRLabelPage,
    parse_label_txt,
    write_file_state_txt,
    write_label_txt,
)

BASE_URL = "https://kimhannom.fit.hcmus.edu.vn"
UPLOAD_ENDPOINT = "/api/web/clc-sinonom/image-upload"
OCR_ENDPOINT = "/api/web/clc-sinonom/image-ocr"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

STATUS_PENDING = "PENDING"
STATUS_UPLOADED = "UPLOADED"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

LOGGER = logging.getLogger("kim_scheduler")


class TokenAuthError(RuntimeError):
    """Raised when the API rejects the Bearer token (HTTP 401/403)."""


class TransientAPIError(RuntimeError):
    """Raised on retryable network, HTTP 429/5xx, or transient API errors."""


@dataclass(frozen=True)
class QuotaConfig:
    """Sliding-window API rate limits."""

    per_minute: int = 25
    per_hour: int = 50
    per_day: int = 200
    per_week: int = 600
    min_delay: float = 2.5

    def windows(self) -> list[tuple[str, int, int]]:
        """Return (label, window_seconds, max_requests) tuples."""
        return [
            ("1m", 60, self.per_minute),
            ("1h", 3600, self.per_hour),
            ("1d", 86400, self.per_day),
            ("7d", 604800, self.per_week),
        ]


@dataclass
class TaskRecord:
    """Represents a single image OCR task in SQLite."""

    id: int
    image_path: str
    status: str
    uploaded_file_name: str | None
    retry_count: int
    last_error: str | None


class StateStore:
    """SQLite persistence for task checkpoints and API request timestamps."""

    def __init__(self, db_path: Path, quota_db_path: Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row

        if (
            quota_db_path is not None
            and Path(quota_db_path).resolve() != self.db_path.resolve()
        ):
            self.quota_db_path = Path(quota_db_path)
            self.quota_db_path.parent.mkdir(parents=True, exist_ok=True)
            self._quota_conn = sqlite3.connect(str(self.quota_db_path), timeout=30.0)
            self._quota_conn.row_factory = sqlite3.Row
        else:
            self.quota_db_path = self.db_path
            self._quota_conn = self._conn

        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    uploaded_file_name TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)"
            )

        with self._quota_conn:
            self._quota_conn.execute("PRAGMA journal_mode=WAL;")
            self._quota_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    endpoint TEXT NOT NULL,
                    status_code INTEGER,
                    token_id TEXT NOT NULL DEFAULT 'default'
                )
                """
            )
            cols = {
                row["name"]
                for row in self._quota_conn.execute(
                    "PRAGMA table_info(request_logs)"
                ).fetchall()
            }
            if "token_id" not in cols:
                self._quota_conn.execute(
                    "ALTER TABLE request_logs ADD COLUMN token_id TEXT NOT NULL DEFAULT 'default'"
                )
            self._quota_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_req_ts ON request_logs(timestamp)"
            )
            self._quota_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_req_token_ts ON request_logs(token_id, timestamp)"
            )

    def close(self) -> None:
        if self._quota_conn is not self._conn:
            self._quota_conn.close()
        self._conn.close()

    def sync_images(self, image_paths: list[Path]) -> int:
        """Register any new images as PENDING. Returns count of newly added tasks."""
        now = time.time()
        added = 0
        with self._conn:
            for path in image_paths:
                cur = self._conn.execute(
                    """
                    INSERT OR IGNORE INTO tasks (
                        image_path, status, retry_count, created_at, updated_at
                    ) VALUES (?, ?, 0, ?, ?)
                    """,
                    (str(path), STATUS_PENDING, now, now),
                )
                added += cur.rowcount
        return added

    def reset_failed_tasks(self) -> int:
        """Reset FAILED tasks back to PENDING so they can be retried."""
        now = time.time()
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE tasks
                SET status = CASE
                        WHEN uploaded_file_name IS NOT NULL AND uploaded_file_name != ''
                        THEN ?
                        ELSE ?
                    END,
                    retry_count = 0,
                    last_error = NULL,
                    updated_at = ?
                WHERE status = ?
                """,
                (STATUS_UPLOADED, STATUS_PENDING, now, STATUS_FAILED),
            )
            return cur.rowcount

    def next_pending_task(self) -> TaskRecord | None:
        """Fetch the next PENDING or UPLOADED task in deterministic path order."""
        row = self._conn.execute(
            """
            SELECT id, image_path, status, uploaded_file_name, retry_count, last_error
            FROM tasks
            WHERE status IN (?, ?)
            ORDER BY image_path ASC
            LIMIT 1
            """,
            (STATUS_PENDING, STATUS_UPLOADED),
        ).fetchone()
        if row is None:
            return None
        return TaskRecord(
            id=int(row["id"]),
            image_path=str(row["image_path"]),
            status=str(row["status"]),
            uploaded_file_name=row["uploaded_file_name"],
            retry_count=int(row["retry_count"]),
            last_error=row["last_error"],
        )

    def mark_uploaded(self, task_id: int, uploaded_file_name: str) -> None:
        now = time.time()
        with self._conn:
            self._conn.execute(
                """
                UPDATE tasks
                SET status = ?, uploaded_file_name = ?, updated_at = ?, last_error = NULL
                WHERE id = ?
                """,
                (STATUS_UPLOADED, uploaded_file_name, now, task_id),
            )

    def mark_completed(self, task_id: int) -> None:
        now = time.time()
        with self._conn:
            self._conn.execute(
                """
                UPDATE tasks
                SET status = ?, updated_at = ?, completed_at = ?, last_error = NULL
                WHERE id = ?
                """,
                (STATUS_COMPLETED, now, now, task_id),
            )

    def mark_retry(self, task_id: int, error_msg: str, failed: bool = False) -> int:
        """Increment retry_count and optionally mark task FAILED. Returns new count."""
        now = time.time()
        with self._conn:
            row = self._conn.execute(
                "SELECT retry_count FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            new_count = (int(row["retry_count"]) if row else 0) + 1
            if failed:
                self._conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, retry_count = ?, last_error = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (STATUS_FAILED, new_count, error_msg, now, task_id),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE tasks
                    SET retry_count = ?, last_error = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (new_count, error_msg, now, task_id),
                )
        return new_count

    def get_task_counts(self) -> dict[str, int]:
        counts = {
            STATUS_PENDING: 0,
            STATUS_UPLOADED: 0,
            STATUS_COMPLETED: 0,
            STATUS_FAILED: 0,
        }
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM tasks GROUP BY status"
        ).fetchall()
        for row in rows:
            counts[str(row["status"])] = int(row["cnt"])
        return counts

    def record_request(
        self,
        endpoint: str,
        status_code: int | None,
        ts: float | None = None,
        token_id: str = "default",
    ) -> None:
        now = time.time() if ts is None else ts
        with self._quota_conn:
            self._quota_conn.execute(
                "INSERT INTO request_logs (timestamp, endpoint, status_code, token_id) VALUES (?, ?, ?, ?)",
                (now, endpoint, status_code, token_id),
            )

    def get_request_timestamps_since(
        self, since_ts: float, token_id: str = "default"
    ) -> list[float]:
        rows = self._quota_conn.execute(
            "SELECT timestamp FROM request_logs WHERE token_id = ? AND timestamp >= ? ORDER BY timestamp ASC",
            (token_id, since_ts),
        ).fetchall()
        return [float(row["timestamp"]) for row in rows]

    def get_last_request_timestamp(self, token_id: str = "default") -> float | None:
        row = self._quota_conn.execute(
            "SELECT MAX(timestamp) AS max_ts FROM request_logs WHERE token_id = ?",
            (token_id,),
        ).fetchone()
        if row is None or row["max_ts"] is None:
            return None
        return float(row["max_ts"])

    def prune_old_requests(self, older_than_seconds: int = 604800) -> int:
        cutoff = time.time() - older_than_seconds
        with self._quota_conn:
            cur = self._quota_conn.execute(
                "DELETE FROM request_logs WHERE timestamp < ?", (cutoff,)
            )
            return cur.rowcount


class RateLimiter:
    """Sliding-window rate limiter backed by SQLite request logs."""

    def __init__(
        self,
        store: StateStore,
        config: QuotaConfig,
        stop_event: threading.Event | None = None,
        token_id: str = "default",
    ) -> None:
        self.store = store
        self.config = config
        self.stop_event = stop_event or threading.Event()
        self.token_id = token_id

    def get_usage_summary(self, now: float | None = None) -> dict[str, tuple[int, int]]:
        """Return {window_label: (used, limit)} for all configured windows."""
        now = time.time() if now is None else now
        max_window = max(w for _, w, _ in self.config.windows())
        timestamps = self.store.get_request_timestamps_since(
            now - max_window, token_id=self.token_id
        )
        summary: dict[str, tuple[int, int]] = {}
        for label, window_sec, limit in self.config.windows():
            cutoff = now - window_sec
            used = sum(1 for ts in timestamps if ts > cutoff)
            summary[label] = (used, limit)
        return summary

    def compute_wait_seconds(
        self, now: float | None = None
    ) -> tuple[float, str | None]:
        """Compute how many seconds to wait before the next request is allowed."""
        now = time.time() if now is None else now
        max_wait = 0.0
        reason: str | None = None

        last_ts = self.store.get_last_request_timestamp(token_id=self.token_id)
        if last_ts is not None and self.config.min_delay > 0:
            elapsed = now - last_ts
            if elapsed < self.config.min_delay:
                delay_wait = self.config.min_delay - elapsed
                if delay_wait > max_wait:
                    max_wait = delay_wait
                    reason = f"min-delay ({self.config.min_delay:.1f}s)"

        max_window = max(w for _, w, _ in self.config.windows())
        timestamps = self.store.get_request_timestamps_since(
            now - max_window, token_id=self.token_id
        )

        for label, window_sec, limit in self.config.windows():
            if limit <= 0:
                continue
            cutoff = now - window_sec
            in_window = [ts for ts in timestamps if ts > cutoff]
            if len(in_window) >= limit:
                # Oldest request that must expire before a slot opens up.
                blocking_ts = in_window[len(in_window) - limit]
                wait = (blocking_ts + window_sec) - now + 0.5
                if wait > max_wait:
                    max_wait = wait
                    reason = f"quota {label} ({len(in_window)}/{limit})"

        return max_wait, reason

    def wait_for_slot(self, allow_long_wait: bool = True) -> bool:
        """Sleep until a request slot is available.

        Returns True if a slot is ready, False if interrupted or blocked when
        ``allow_long_wait`` is False.
        """
        while not self.stop_event.is_set():
            wait_sec, reason = self.compute_wait_seconds()
            if wait_sec <= 0:
                return True

            if not allow_long_wait and wait_sec > max(self.config.min_delay, 5.0):
                LOGGER.warning(
                    "Quota reached (%s). Exiting because --no-wait-quota is set (next slot in %.1fs).",
                    reason,
                    wait_sec,
                )
                return False

            if wait_sec >= 10.0:
                resume_at = datetime.fromtimestamp(time.time() + wait_sec).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                LOGGER.info(
                    "Rate limit reached (%s). Sleeping %.1fs until %s...",
                    reason,
                    wait_sec,
                    resume_at,
                )
            else:
                LOGGER.debug("Throttle sleep %.2fs (%s)", wait_sec, reason)

            # Sleep in small chunks so SIGINT/SIGTERM wakes immediately.
            remaining = wait_sec
            while remaining > 0 and not self.stop_event.is_set():
                chunk = min(remaining, 2.0)
                if self.stop_event.wait(chunk):
                    return False
                remaining -= chunk

        return False


class KimHanNomClient:
    """HTTP client for Kim Hán Nôm upload and OCR endpoints."""

    def __init__(
        self,
        token: str,
        user_agent: str = DEFAULT_USER_AGENT,
        base_url: str = BASE_URL,
        timeout: int = 180,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "User-Agent": user_agent,
                "Accept": "application/json",
            }
        )

    @staticmethod
    def _parse_envelope(response: requests.Response, step_name: str) -> dict[str, Any]:
        if response.status_code in (401, 403):
            raise TokenAuthError(
                f"{step_name}: Authentication failed (HTTP {response.status_code}). "
                "Please refresh TOKEN in .env."
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientAPIError(
                f"{step_name}: HTTP {response.status_code} - {response.text[:300]}"
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"{step_name}: HTTP error {exc}") from exc

        try:
            envelope = response.json()
        except ValueError as exc:
            raise TransientAPIError(
                f"{step_name}: Non-JSON response: {response.text[:300]}"
            ) from exc

        if envelope.get("is_success") is not True or envelope.get("code") != "000000":
            msg = envelope.get("message", "")
            code = envelope.get("code", "")
            raise RuntimeError(f"{step_name} API error: code={code!r}, message={msg!r}")

        data = envelope.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"{step_name}: 'data' field is not an object")
        return data

    def upload_image(self, image_path: Path) -> tuple[str, int]:
        """Upload an image and return ``(remote_file_name, http_status)``."""
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            mime_type = "image/jpeg"

        try:
            with image_path.open("rb") as fh:
                response = self.session.post(
                    f"{self.base_url}{UPLOAD_ENDPOINT}",
                    files={"image_file": (image_path.name, fh, mime_type)},
                    timeout=self.timeout,
                )
        except requests.RequestException as exc:
            raise TransientAPIError(f"image-upload network error: {exc}") from exc

        data = self._parse_envelope(response, "image-upload")
        file_name = data.get("file_name")
        if not file_name or not isinstance(file_name, str):
            raise RuntimeError(f"image-upload missing file_name in response: {data}")
        return file_name, response.status_code

    def ocr_image(
        self,
        remote_file_name: str,
        ocr_id: int = 1,
        lang_type: int = 2,
        reading_direction: int = 1,
    ) -> tuple[dict[str, Any], int]:
        """Run OCR on an uploaded image and return ``(ocr_data, http_status)``."""
        payload = {
            "ocr_id": ocr_id,
            "lang_type": lang_type,
            "reading_direction": reading_direction,
            "file_name": remote_file_name,
        }
        try:
            response = self.session.post(
                f"{self.base_url}{OCR_ENDPOINT}",
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise TransientAPIError(f"image-ocr network error: {exc}") from exc

        data = self._parse_envelope(response, "image-ocr")
        return data, response.status_code


def extract_han_nom_lines(ocr_data: dict[str, Any]) -> list[str]:
    """Extract Hán Nôm text lines from ``result_ocr_text``."""
    raw = ocr_data.get("result_ocr_text")
    if isinstance(raw, list):
        return [str(line) for line in raw]
    if isinstance(raw, str):
        return raw.splitlines()
    return []


def _coerce_points(raw_points: Any) -> list[list[int]] | None:
    """Validate and coerce a 4-point polygon into ``[[int, int], ...]``."""
    if not isinstance(raw_points, list) or len(raw_points) != 4:
        return None
    coerced: list[list[int]] = []
    for pt in raw_points:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            return None
        try:
            coerced.append([int(round(float(pt[0]))), int(round(float(pt[1])))])
        except (TypeError, ValueError):
            return None
    return coerced


def extract_ppocr_annotations(ocr_data: dict[str, Any]) -> list[PPOCRLabelAnnotation]:
    """Build :class:`PPOCRLabelAnnotation` items from ``details.details`` or ``result_bbox``."""
    annotations: list[PPOCRLabelAnnotation] = []

    details_obj = ocr_data.get("details")
    detail_items = details_obj.get("details") if isinstance(details_obj, dict) else None

    if isinstance(detail_items, list) and detail_items:
        for item in detail_items:
            if not isinstance(item, dict):
                continue
            text = (
                unicodedata.normalize("NFC", str(item.get("transcription", "")))
                .replace("\n", " ")
                .strip()
            )
            points = _coerce_points(item.get("points"))
            if text and points is not None:
                annotations.append(
                    PPOCRLabelAnnotation(transcription=text, points=points)
                )
        if annotations:
            return annotations

    # Fallback to result_bbox: [ [[[x1,y1],...], [text, score]], ... ]
    result_bbox = ocr_data.get("result_bbox")
    if isinstance(result_bbox, list):
        for entry in result_bbox:
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            points = _coerce_points(entry[0])
            text_part = entry[1]
            raw_text = (
                text_part[0]
                if isinstance(text_part, (list, tuple)) and text_part
                else text_part
            )
            text = (
                unicodedata.normalize("NFC", str(raw_text)).replace("\n", " ").strip()
            )
            if text and points is not None:
                annotations.append(
                    PPOCRLabelAnnotation(transcription=text, points=points)
                )

    return annotations


def _atomic_write_text(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, target)


def save_ocr_outputs(
    image_path: Path,
    ocr_data: dict[str, Any],
    output_dir: Path,
    ppocr_prefix: str = "ch",
) -> tuple[Path, Path]:
    """Save raw JSON, Hán Nôm plain text, and update ``Label.txt`` & ``fileState.txt``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # 1. Raw JSON response
    json_path = output_dir / f"{stem}_response.json"
    _atomic_write_text(json_path, json.dumps(ocr_data, ensure_ascii=False, indent=2))

    # 2. Hán Nôm plain text
    lines = extract_han_nom_lines(ocr_data)
    txt_path = output_dir / f"{stem}_han_nom.txt"
    _atomic_write_text(txt_path, "\n".join(lines))

    # 3. PPOCRLabel Label.txt & fileState.txt
    label_path = output_dir / "Label.txt"
    state_path = output_dir / "fileState.txt"

    existing_pages: dict[str, PPOCRLabelPage] = {}
    if label_path.exists():
        for page in parse_label_txt(label_path):
            existing_pages[page.image_path] = page

    rel_image_path = (
        f"{ppocr_prefix.strip('/')}/{image_path.name}"
        if ppocr_prefix.strip("/")
        else image_path.name
    )
    annotations = extract_ppocr_annotations(ocr_data)
    existing_pages[rel_image_path] = PPOCRLabelPage(
        image_path=rel_image_path,
        annotations=annotations,
    )

    ordered_pages = [existing_pages[k] for k in sorted(existing_pages)]
    tmp_label = output_dir / ".Label.txt.tmp"
    tmp_state = output_dir / ".fileState.txt.tmp"
    write_label_txt(ordered_pages, tmp_label, overwrite=True)
    write_file_state_txt(ordered_pages, tmp_state, overwrite=True)
    os.replace(tmp_label, label_path)
    os.replace(tmp_state, state_path)

    return json_path, txt_path


def discover_images(input_dir: Path) -> list[Path]:
    """Find all supported image files in ``input_dir`` sorted by filename."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    images = [
        p.resolve()
        for p in sorted(input_dir.iterdir(), key=lambda p: p.name)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return images


def compute_token_id(token: str) -> str:
    """Return a short deterministic SHA-256 identifier for a Bearer token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def resolve_token(cli_token: str | None, interactive: bool = True) -> str:
    """Resolve Bearer token from CLI arg, .env/environment, or interactive prompt."""
    load_dotenv()
    token = (cli_token or os.getenv("TOKEN") or "").strip()
    if not token and interactive and sys.stdin.isatty():
        token = getpass.getpass("Nhập Kim API Bearer token: ").strip()
    if not token:
        raise ValueError(
            "Missing API token. Set TOKEN in .env, pass --token, or run interactively."
        )
    return token


def setup_logging(log_file: Path | None, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


def print_status(store: StateStore, limiter: RateLimiter) -> None:
    counts = store.get_task_counts()
    usage = limiter.get_usage_summary()
    total = sum(counts.values())
    print("=== Kim Hán Nôm OCR Scheduler Status ===")
    print(
        f"Tasks: total={total}, "
        f"completed={counts[STATUS_COMPLETED]}, "
        f"uploaded={counts[STATUS_UPLOADED]}, "
        f"pending={counts[STATUS_PENDING]}, "
        f"failed={counts[STATUS_FAILED]}"
    )
    quota_str = ", ".join(
        f"{label}: {used}/{limit}" for label, (used, limit) in usage.items()
    )
    print(f"Quota usage: {quota_str}")


def run_scheduler(
    store: StateStore,
    limiter: RateLimiter,
    client: KimHanNomClient | None,
    input_dir: Path,
    output_dir: Path,
    *,
    ocr_id: int = 1,
    lang_type: int = 2,
    reading_direction: int = 1,
    ppocr_prefix: str = "ch",
    max_retries: int = 5,
    daemon: bool = False,
    poll_interval: int = 60,
    allow_long_wait: bool = True,
    stop_event: threading.Event | None = None,
) -> int:
    """Execute the scheduler loop. Returns exit code (0 on success, 1 on auth/fatal error)."""
    stop_event = stop_event or threading.Event()

    while not stop_event.is_set():
        images = discover_images(input_dir)
        added = store.sync_images(images)
        if added > 0:
            LOGGER.info("Discovered %d new image(s) in %s.", added, input_dir)

        store.prune_old_requests()
        task = store.next_pending_task()

        if task is None:
            if not daemon:
                LOGGER.info("All tasks processed. Exiting.")
                print_status(store, limiter)
                return 0
            LOGGER.info(
                "No pending tasks. Daemon sleeping %ds before re-scanning %s...",
                poll_interval,
                input_dir,
            )
            if stop_event.wait(poll_interval):
                break
            continue

        assert client is not None
        image_path = Path(task.image_path)
        if not image_path.is_file():
            err = f"Image file missing on disk: {image_path}"
            LOGGER.error(err)
            store.mark_retry(task.id, err, failed=True)
            continue

        try:
            # Step 1: Upload (skip if already uploaded in a previous interrupted run)
            remote_name = task.uploaded_file_name
            if task.status == STATUS_PENDING or not remote_name:
                if not limiter.wait_for_slot(allow_long_wait=allow_long_wait):
                    return 0
                LOGGER.info("[Upload] %s", image_path.name)
                try:
                    remote_name, status_code = client.upload_image(image_path)
                    store.record_request(
                        UPLOAD_ENDPOINT, status_code, token_id=limiter.token_id
                    )
                except TransientAPIError as exc:
                    store.record_request(
                        UPLOAD_ENDPOINT, None, token_id=limiter.token_id
                    )
                    raise exc
                store.mark_uploaded(task.id, remote_name)
                LOGGER.info("[Upload OK] %s -> %s", image_path.name, remote_name)

            # Step 2: OCR
            if not limiter.wait_for_slot(allow_long_wait=allow_long_wait):
                return 0
            LOGGER.info("[OCR] %s (remote=%s)", image_path.name, remote_name)
            try:
                ocr_data, status_code = client.ocr_image(
                    remote_name,
                    ocr_id=ocr_id,
                    lang_type=lang_type,
                    reading_direction=reading_direction,
                )
                store.record_request(
                    OCR_ENDPOINT, status_code, token_id=limiter.token_id
                )
            except TransientAPIError as exc:
                store.record_request(OCR_ENDPOINT, None, token_id=limiter.token_id)
                raise exc

            json_path, txt_path = save_ocr_outputs(
                image_path=image_path,
                ocr_data=ocr_data,
                output_dir=output_dir,
                ppocr_prefix=ppocr_prefix,
            )
            store.mark_completed(task.id)
            LOGGER.info(
                "[Done] %s -> %s, %s",
                image_path.name,
                json_path.name,
                txt_path.name,
            )

        except TokenAuthError as exc:
            LOGGER.error("Authentication stopped scheduler: %s", exc)
            return 1

        except TransientAPIError as exc:
            new_count = store.mark_retry(
                task.id, str(exc), failed=(task.retry_count + 1 >= max_retries)
            )
            if new_count >= max_retries:
                LOGGER.error(
                    "[Failed] %s exceeded max retries (%d): %s",
                    image_path.name,
                    max_retries,
                    exc,
                )
            else:
                backoff = min(300.0, 5.0 * (2 ** (new_count - 1)))
                if "429" in str(exc):
                    backoff = max(backoff, 60.0)
                LOGGER.warning(
                    "[Retry %d/%d] %s: %s. Backing off %.1fs...",
                    new_count,
                    max_retries,
                    image_path.name,
                    exc,
                    backoff,
                )
                if stop_event.wait(backoff):
                    break

        except Exception as exc:  # noqa: BLE001
            new_count = store.mark_retry(
                task.id, str(exc), failed=(task.retry_count + 1 >= max_retries)
            )
            LOGGER.error(
                "[Error %d/%d] %s: %s",
                new_count,
                max_retries,
                image_path.name,
                exc,
            )

    LOGGER.info("Shutdown signal received. Exiting cleanly.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scheduled Kim Hán Nôm OCR runner with SQLite checkpointing and rate limiting."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "pilot" / "images" / "han_nom",
        help="Directory containing Sino-Nom page images (.jpg/.jpeg/.png).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "pilot" / "api_output",
        help="Directory to write JSON, text, Label.txt, and fileState.txt outputs.",
    )
    parser.add_argument(
        "--state-db",
        type=Path,
        default=None,
        help="Path to SQLite task state DB (default: <output-dir>/scheduler_state.db).",
    )
    parser.add_argument(
        "--quota-db",
        type=Path,
        default=Path.home() / ".cache" / "kimhanom_quota.db",
        help="Path to shared SQLite quota DB per token (default: ~/.cache/kimhanom_quota.db).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file (default: <output-dir>/scheduler.log).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Bearer JWT token (overrides TOKEN in .env).",
    )
    parser.add_argument(
        "--ppocr-prefix",
        type=str,
        default="ch",
        help="Folder prefix used in Label.txt image paths (default: ch).",
    )
    parser.add_argument(
        "--ocr-id", type=int, default=1, help="API ocr_id (default: 1)."
    )
    parser.add_argument(
        "--lang-type", type=int, default=2, help="API lang_type (default: 2)."
    )
    parser.add_argument(
        "--reading-direction",
        type=int,
        default=1,
        help="API reading_direction (default: 1).",
    )
    parser.add_argument(
        "--limit-minute", type=int, default=25, help="Max requests per minute."
    )
    parser.add_argument(
        "--limit-hour", type=int, default=50, help="Max requests per hour."
    )
    parser.add_argument(
        "--limit-day", type=int, default=200, help="Max requests per day."
    )
    parser.add_argument(
        "--limit-week", type=int, default=600, help="Max requests per week."
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=2.5,
        help="Minimum seconds between consecutive API requests (default: 2.5).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per image before marking FAILED (default: 5).",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Keep running and poll input-dir for new images after queue finishes.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between directory scans in --daemon mode (default: 60).",
    )
    parser.add_argument(
        "--no-wait-quota",
        action="store_true",
        help="Exit immediately instead of sleeping when hourly/daily/weekly quota is hit.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reset previously FAILED tasks back to PENDING/UPLOADED before running.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current task and quota status from the DB and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan input-dir, update DB task queue, print status, and exit without API calls.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    state_db: Path = args.state_db or (output_dir / "scheduler_state.db")
    log_file: Path = args.log_file or (output_dir / "scheduler.log")

    setup_logging(log_file=log_file, verbose=args.verbose)

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        LOGGER.info("Received signal %s; stopping after current step...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    quota_cfg = QuotaConfig(
        per_minute=args.limit_minute,
        per_hour=args.limit_hour,
        per_day=args.limit_day,
        per_week=args.limit_week,
        min_delay=args.min_delay,
    )

    store = StateStore(state_db, quota_db_path=args.quota_db)

    # Resolve token_id early if available so --status/--dry-run show that token's quota.
    load_dotenv()
    raw_token_for_id = (args.token or os.getenv("TOKEN") or "").strip()
    token_id = compute_token_id(raw_token_for_id) if raw_token_for_id else "default"
    limiter = RateLimiter(
        store=store, config=quota_cfg, stop_event=stop_event, token_id=token_id
    )

    try:
        if args.retry_failed:
            reset_count = store.reset_failed_tasks()
            LOGGER.info("Reset %d failed task(s).", reset_count)

        if args.status:
            print_status(store, limiter)
            return 0

        if args.dry_run:
            images = discover_images(args.input_dir)
            added = store.sync_images(images)
            LOGGER.info(
                "Dry-run: scanned %d image(s), added %d new.", len(images), added
            )
            print_status(store, limiter)
            return 0

        token = resolve_token(args.token, interactive=not args.daemon)
        limiter.token_id = compute_token_id(token)
        client = KimHanNomClient(token=token)

        return run_scheduler(
            store=store,
            limiter=limiter,
            client=client,
            input_dir=args.input_dir,
            output_dir=output_dir,
            ocr_id=args.ocr_id,
            lang_type=args.lang_type,
            reading_direction=args.reading_direction,
            ppocr_prefix=args.ppocr_prefix,
            max_retries=args.max_retries,
            daemon=args.daemon,
            poll_interval=args.poll_interval,
            allow_long_wait=not args.no_wait_quota,
            stop_event=stop_event,
        )
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
