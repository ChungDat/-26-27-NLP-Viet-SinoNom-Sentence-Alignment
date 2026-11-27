"""Unit tests for scripts/kim_api/scheduler.py."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kim_api import scheduler  # noqa: E402
from src.ppocr import parse_label_txt  # noqa: E402


SAMPLE_OCR_DATA = {
    "result_file_name": "temp_abc.jpg",
    "result_ocr_text": ["傳各聖", "課初"],
    "result_bbox": [
        [[[423.0, 401.0], [469.0, 401.0], [466.0, 680.0], [421.0, 680.0]], ["傳各聖", 0.99]],
    ],
    "details": {
        "file_name": "",
        "details": [
            {
                "transcription": "傳各聖",
                "points": [[423, 401], [469, 401], [466, 680], [421, 680]],
                "difficult": False,
            },
            {
                "transcription": "課初",
                "points": [[289, 92], [322, 92], [322, 987], [289, 987]],
                "difficult": False,
            },
        ],
    },
}


class TestStateStoreAndRateLimiter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = scheduler.StateStore(self.root / "state.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_sync_and_lifecycle(self) -> None:
        img1 = self.root / "001.jpg"
        img2 = self.root / "002.jpg"
        img1.write_bytes(b"fake")
        img2.write_bytes(b"fake")

        self.assertEqual(self.store.sync_images([img1, img2]), 2)
        self.assertEqual(self.store.sync_images([img1, img2]), 0)

        t1 = self.store.next_pending_task()
        assert t1 is not None
        self.assertEqual(Path(t1.image_path), img1)
        self.assertEqual(t1.status, scheduler.STATUS_PENDING)

        self.store.mark_uploaded(t1.id, "remote_001.jpg")
        t1_again = self.store.next_pending_task()
        assert t1_again is not None
        self.assertEqual(t1_again.status, scheduler.STATUS_UPLOADED)
        self.assertEqual(t1_again.uploaded_file_name, "remote_001.jpg")

        self.store.mark_completed(t1.id)
        t2 = self.store.next_pending_task()
        assert t2 is not None
        self.assertEqual(Path(t2.image_path), img2)

        self.store.mark_retry(t2.id, "boom", failed=True)
        self.assertIsNone(self.store.next_pending_task())
        counts = self.store.get_task_counts()
        self.assertEqual(counts[scheduler.STATUS_COMPLETED], 1)
        self.assertEqual(counts[scheduler.STATUS_FAILED], 1)

        reset = self.store.reset_failed_tasks()
        self.assertEqual(reset, 1)
        t2_reset = self.store.next_pending_task()
        assert t2_reset is not None
        self.assertEqual(t2_reset.status, scheduler.STATUS_PENDING)
        self.assertEqual(t2_reset.retry_count, 0)

    def test_rate_limiter_sliding_window(self) -> None:
        cfg = scheduler.QuotaConfig(
            per_minute=2, per_hour=10, per_day=20, per_week=50, min_delay=1.0
        )
        limiter = scheduler.RateLimiter(self.store, cfg)

        now = time.time()
        self.store.record_request("/upload", 200, ts=now - 20.0)
        self.store.record_request("/ocr", 200, ts=now - 10.0)

        wait, reason = limiter.compute_wait_seconds(now=now)
        self.assertIsNotNone(reason)
        self.assertIn("1m", reason or "")
        # Oldest in 1m window was at now-20, so expires at now+40 (+0.5s buffer).
        self.assertAlmostEqual(wait, 40.5, places=1)

        # When allow_long_wait=False, wait_for_slot returns False immediately.
        self.assertFalse(limiter.wait_for_slot(allow_long_wait=False))


class TestOutputsAndScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.in_dir = self.root / "in"
        self.out_dir = self.root / "out"
        self.in_dir.mkdir()
        self.out_dir.mkdir()
        self.store = scheduler.StateStore(self.out_dir / "state.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_save_ocr_outputs_and_ppocr_fallback(self) -> None:
        img = self.in_dir / "book.010.jpg"
        img.write_bytes(b"jpg")

        # Also test fallback to result_bbox when details.details is absent
        fallback_data = {
            "result_ocr_text": ["傳各聖"],
            "result_bbox": SAMPLE_OCR_DATA["result_bbox"],
        }
        json_path, txt_path = scheduler.save_ocr_outputs(
            img, fallback_data, self.out_dir, ppocr_prefix="ch"
        )
        self.assertTrue(json_path.is_file())
        self.assertTrue(txt_path.is_file())
        self.assertEqual(txt_path.read_text(encoding="utf-8"), "傳各聖")

        pages = parse_label_txt(self.out_dir / "Label.txt")
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].image_path, "ch/book.010.jpg")
        self.assertEqual(pages[0].annotations[0].transcription, "傳各聖")
        self.assertEqual(
            (self.out_dir / "fileState.txt").read_text(encoding="utf-8").strip(),
            "ch/book.010.jpg\t1",
        )

    def test_run_scheduler_resumes_uploaded_task(self) -> None:
        img1 = self.in_dir / "book.001.jpg"
        img2 = self.in_dir / "book.002.jpg"
        img1.write_bytes(b"1")
        img2.write_bytes(b"2")

        # Pre-seed img1 as already UPLOADED
        self.store.sync_images([img1, img2])
        t1 = self.store.next_pending_task()
        assert t1 is not None
        self.store.mark_uploaded(t1.id, "pre_uploaded_001.jpg")

        client = MagicMock(spec=scheduler.KimHanNomClient)
        client.upload_image.return_value = ("uploaded_002.jpg", 200)
        client.ocr_image.return_value = (SAMPLE_OCR_DATA, 200)

        cfg = scheduler.QuotaConfig(min_delay=0.0)
        limiter = scheduler.RateLimiter(self.store, cfg)

        rc = scheduler.run_scheduler(
            store=self.store,
            limiter=limiter,
            client=client,
            input_dir=self.in_dir,
            output_dir=self.out_dir,
            daemon=False,
        )
        self.assertEqual(rc, 0)
        # img1 skipped upload; only img2 was uploaded
        client.upload_image.assert_called_once_with(img2.resolve())
        self.assertEqual(client.ocr_image.call_count, 2)

        counts = self.store.get_task_counts()
        self.assertEqual(counts[scheduler.STATUS_COMPLETED], 2)
        pages = parse_label_txt(self.out_dir / "Label.txt")
        self.assertEqual([p.image_path for p in pages], ["ch/book.001.jpg", "ch/book.002.jpg"])

    def test_run_scheduler_stops_on_auth_error(self) -> None:
        img = self.in_dir / "book.001.jpg"
        img.write_bytes(b"1")

        client = MagicMock(spec=scheduler.KimHanNomClient)
        client.upload_image.side_effect = scheduler.TokenAuthError("401 expired")

        limiter = scheduler.RateLimiter(
            self.store, scheduler.QuotaConfig(min_delay=0.0), threading.Event()
        )
        rc = scheduler.run_scheduler(
            store=self.store,
            limiter=limiter,
            client=client,
            input_dir=self.in_dir,
            output_dir=self.out_dir,
        )
        self.assertEqual(rc, 1)
        counts = self.store.get_task_counts()
        self.assertEqual(counts[scheduler.STATUS_PENDING], 1)
        self.assertEqual(counts[scheduler.STATUS_FAILED], 0)

    def test_per_token_quota_isolation(self) -> None:
        cfg = scheduler.QuotaConfig(
            per_minute=2, per_hour=10, per_day=20, per_week=50, min_delay=0.0
        )
        tok_a = scheduler.compute_token_id("token-alpha")
        tok_b = scheduler.compute_token_id("token-beta")
        lim_a = scheduler.RateLimiter(self.store, cfg, token_id=tok_a)
        lim_b = scheduler.RateLimiter(self.store, cfg, token_id=tok_b)

        now = time.time()
        self.store.record_request("/upload", 200, ts=now - 10.0, token_id=tok_a)
        self.store.record_request("/ocr", 200, ts=now - 5.0, token_id=tok_a)

        wait_a, reason_a = lim_a.compute_wait_seconds(now=now)
        wait_b, reason_b = lim_b.compute_wait_seconds(now=now)
        self.assertGreater(wait_a, 0.0)
        self.assertIn("1m", reason_a or "")
        self.assertEqual(wait_b, 0.0)
        self.assertIsNone(reason_b)

    def test_shared_quota_db_across_folders(self) -> None:
        quota_db = self.root / "shared_quota.db"
        store1 = scheduler.StateStore(self.root / "folder1.db", quota_db_path=quota_db)
        store2 = scheduler.StateStore(self.root / "folder2.db", quota_db_path=quota_db)
        try:
            tok = scheduler.compute_token_id("shared-token")
            now = time.time()
            store1.record_request("/upload", 200, ts=now - 5.0, token_id=tok)
            store1.record_request("/ocr", 200, ts=now - 2.0, token_id=tok)

            cfg = scheduler.QuotaConfig(
                per_minute=2, per_hour=10, per_day=20, per_week=50, min_delay=0.0
            )
            lim2 = scheduler.RateLimiter(store2, cfg, token_id=tok)
            wait, reason = lim2.compute_wait_seconds(now=now)
            self.assertGreater(wait, 0.0)
            self.assertIn("1m", reason or "")
        finally:
            store1.close()
            store2.close()


if __name__ == "__main__":
    unittest.main()
