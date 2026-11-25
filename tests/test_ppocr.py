"""Unit tests for src/ppocr.py — PPOCRLabel models, conversion, IO."""

from __future__ import annotations

import json
import tempfile
import unicodedata
from pathlib import Path

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ppocr import (
    PPOCRLabelAnnotation,
    PPOCRLabelPage,
    bbox_to_points,
    make_image_path,
    migrate_ocr_document_json,
    normalize_transcription,
    ocr_page_to_ppocr,
    parse_header_paragraph,
    parse_label_txt,
    write_file_state_txt,
    write_label_txt,
)
from src.schemas import OCRPage, OCRParagraph

import unittest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page(
    paragraphs: list[OCRParagraph],
    pdf_page: int = 1,
    doc_page: int = 1,
    method: str = "tesseract",
) -> OCRPage:
    return OCRPage(
        pdf_page_number=pdf_page,
        document_page_number=doc_page,
        page_name=f"{pdf_page:03d}",
        method=method,
        paragraphs=paragraphs,
    )


def _make_para(
    text: str,
    bbox: tuple[int, int, int, int] | None = (10, 20, 100, 50),
    pid: int = 1,
) -> OCRParagraph:
    return OCRParagraph(paragraph_id=pid, bbox=bbox, text=text)


# ---------------------------------------------------------------------------
# bbox_to_points
# ---------------------------------------------------------------------------


class TestBboxToPoints(unittest.TestCase):
    def test_basic(self) -> None:
        result = bbox_to_points((10, 20, 30, 40))
        self.assertEqual(result, [[10, 20], [30, 20], [30, 40], [10, 40]])

    def test_zero_area(self) -> None:
        result = bbox_to_points((5, 5, 5, 5))
        self.assertEqual(result, [[5, 5], [5, 5], [5, 5], [5, 5]])

    def test_top_left_origin(self) -> None:
        result = bbox_to_points((0, 0, 100, 200))
        self.assertEqual(result, [[0, 0], [100, 0], [100, 200], [0, 200]])

    def test_returns_four_points(self) -> None:
        self.assertEqual(len(bbox_to_points((1, 2, 3, 4))), 4)

    def test_each_point_is_two_ints(self) -> None:
        for pt in bbox_to_points((1, 2, 3, 4)):
            self.assertEqual(len(pt), 2)
            self.assertIsInstance(pt[0], int)
            self.assertIsInstance(pt[1], int)


# ---------------------------------------------------------------------------
# make_image_path
# ---------------------------------------------------------------------------


class TestMakeImagePath(unittest.TestCase):
    def test_no_index(self) -> None:
        self.assertEqual(make_image_path("doc", "vi", 1), "vi/doc.001.jpg")

    def test_with_index(self) -> None:
        self.assertEqual(make_image_path("doc", "vi", 1, index=2), "vi/doc.001_2.jpg")

    def test_large_page_number(self) -> None:
        self.assertEqual(make_image_path("my_doc", "ch", 42), "ch/my_doc.042.jpg")

    def test_zero_index_included(self) -> None:
        # index=0 is a valid sub-index; must appear in path
        path = make_image_path("d", "vi", 3, index=0)
        self.assertIn("_0", path)


# ---------------------------------------------------------------------------
# PPOCRLabelAnnotation validation
# ---------------------------------------------------------------------------


class TestPPOCRLabelAnnotationValidation(unittest.TestCase):
    def _valid_ann(self, **kwargs: object) -> PPOCRLabelAnnotation:
        defaults = {
            "transcription": "hello",
            "points": [[0, 0], [10, 0], [10, 5], [0, 5]],
        }
        defaults.update(kwargs)
        return PPOCRLabelAnnotation(**defaults)  # type: ignore[arg-type]

    def test_valid(self) -> None:
        ann = self._valid_ann()
        self.assertEqual(ann.transcription, "hello")
        self.assertEqual(len(ann.points), 4)

    def test_nfc_applied(self) -> None:
        # Decomposed 'ê' (NFD) should be normalized to composed form (NFC)
        nfd = unicodedata.normalize("NFD", "\u00ea")
        ann = self._valid_ann(transcription=nfd)
        self.assertEqual(ann.transcription, unicodedata.normalize("NFC", nfd))

    def test_embedded_newline_stripped(self) -> None:
        ann = self._valid_ann(transcription="line1\nline2")
        self.assertNotIn("\n", ann.transcription)

    def test_empty_transcription_raises(self) -> None:
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self._valid_ann(transcription="")

    def test_whitespace_only_transcription_raises(self) -> None:
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self._valid_ann(transcription="   ")

    def test_too_few_points_raises(self) -> None:
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self._valid_ann(points=[[0, 0], [10, 0], [10, 5]])

    def test_too_many_points_raises(self) -> None:
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self._valid_ann(points=[[0, 0], [1, 0], [1, 1], [0, 1], [0, 2]])

    def test_non_integer_coords_raises(self) -> None:
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self._valid_ann(points=[[0.5, 0], [1, 0], [1, 1], [0, 1]])

    def test_align_to_optional(self) -> None:
        ann = self._valid_ann(align_to=5)
        self.assertEqual(ann.align_to, 5)
        ann2 = self._valid_ann()
        self.assertIsNone(ann2.align_to)


# ---------------------------------------------------------------------------
# ocr_page_to_ppocr
# ---------------------------------------------------------------------------


class TestOcrPageToPpocr(unittest.TestCase):
    def test_basic_two_paragraphs(self) -> None:
        page = _make_page([
            _make_para("Hello world", bbox=(0, 0, 100, 50), pid=1),
            _make_para("Second para", bbox=(0, 60, 100, 110), pid=2),
        ])
        result = ocr_page_to_ppocr(page, "vi/doc.001.jpg")
        self.assertEqual(result.image_path, "vi/doc.001.jpg")
        self.assertEqual(len(result.annotations), 2)
        self.assertEqual(result.annotations[0].transcription, "Hello world")
        self.assertEqual(result.annotations[1].transcription, "Second para")

    def test_missing_bbox_skipped(self) -> None:
        page = _make_page([
            _make_para("Has bbox", bbox=(0, 0, 100, 50), pid=1),
            _make_para("No bbox", bbox=None, pid=2),
        ])
        result = ocr_page_to_ppocr(page, "vi/doc.001.jpg")
        self.assertEqual(len(result.annotations), 1)
        self.assertEqual(result.annotations[0].transcription, "Has bbox")

    def test_empty_page_gives_empty_annotations(self) -> None:
        page = _make_page([])
        result = ocr_page_to_ppocr(page, "vi/doc.001.jpg")
        self.assertEqual(result.annotations, [])

    def test_bbox_converted_to_four_points(self) -> None:
        page = _make_page([_make_para("text", bbox=(10, 20, 30, 40))])
        result = ocr_page_to_ppocr(page, "vi/doc.001.jpg")
        self.assertEqual(result.annotations[0].points, [[10, 20], [30, 20], [30, 40], [10, 40]])

    def test_align_to_attached(self) -> None:
        page = _make_page([_make_para("text", bbox=(0, 0, 10, 10))], doc_page=7)
        result = ocr_page_to_ppocr(page, "vi/doc.007.jpg", align_to=7)
        self.assertEqual(result.annotations[0].align_to, 7)

    def test_nfc_normalization_applied(self) -> None:
        nfd_text = unicodedata.normalize("NFD", "ph\u00e2n")  # 'phân' in NFD
        page = _make_page([_make_para(nfd_text, bbox=(0, 0, 50, 20))])
        result = ocr_page_to_ppocr(page, "vi/doc.001.jpg")
        self.assertEqual(
            result.annotations[0].transcription,
            unicodedata.normalize("NFC", nfd_text),
        )

    def test_paragraph_order_preserved(self) -> None:
        page = _make_page([
            _make_para("First", bbox=(0, 0, 10, 10), pid=1),
            _make_para("Second", bbox=(0, 20, 10, 30), pid=2),
            _make_para("Third", bbox=(0, 40, 10, 50), pid=3),
        ])
        result = ocr_page_to_ppocr(page, "vi/doc.001.jpg")
        texts = [a.transcription for a in result.annotations]
        self.assertEqual(texts, ["First", "Second", "Third"])


# ---------------------------------------------------------------------------
# write_label_txt / parse_label_txt round-trip
# ---------------------------------------------------------------------------


class TestWriteAndParseLabelTxt(unittest.TestCase):
    def _sample_pages(self) -> list[PPOCRLabelPage]:
        return [
            PPOCRLabelPage(
                image_path="vi/doc.001.jpg",
                annotations=[
                    PPOCRLabelAnnotation(
                        transcription="Hello",
                        points=[[0, 0], [10, 0], [10, 5], [0, 5]],
                        align_to=1,
                    ),
                    PPOCRLabelAnnotation(
                        transcription="World",
                        points=[[0, 10], [10, 10], [10, 15], [0, 15]],
                    ),
                ],
            ),
            PPOCRLabelPage(
                image_path="vi/doc.002.jpg",
                annotations=[
                    PPOCRLabelAnnotation(
                        transcription="Thứ hai",
                        points=[[0, 0], [50, 0], [50, 20], [0, 20]],
                        align_to=2,
                    ),
                ],
            ),
        ]

    def test_round_trip(self) -> None:
        pages = self._sample_pages()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "Label.txt"
            write_label_txt(pages, out)
            parsed = parse_label_txt(out)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].image_path, "vi/doc.001.jpg")
        self.assertEqual(len(parsed[0].annotations), 2)
        self.assertEqual(parsed[0].annotations[0].transcription, "Hello")
        self.assertEqual(parsed[0].annotations[0].align_to, 1)
        self.assertIsNone(parsed[0].annotations[1].align_to)
        self.assertEqual(parsed[1].annotations[0].transcription, "Thứ hai")

    def test_points_preserved(self) -> None:
        pages = self._sample_pages()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "Label.txt"
            write_label_txt(pages, out)
            parsed = parse_label_txt(out)
        self.assertEqual(parsed[0].annotations[0].points, [[0, 0], [10, 0], [10, 5], [0, 5]])

    def test_overwrite_raises_by_default(self) -> None:
        pages = self._sample_pages()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "Label.txt"
            write_label_txt(pages, out)
            with self.assertRaises(FileExistsError):
                write_label_txt(pages, out)

    def test_overwrite_true_succeeds(self) -> None:
        pages = self._sample_pages()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "Label.txt"
            write_label_txt(pages, out)
            write_label_txt(pages, out, overwrite=True)  # should not raise

    def test_utf8_non_ascii(self) -> None:
        pages = [
            PPOCRLabelPage(
                image_path="vi/doc.001.jpg",
                annotations=[
                    PPOCRLabelAnnotation(
                        transcription="phân tích",
                        points=[[0, 0], [1, 0], [1, 1], [0, 1]],
                    )
                ],
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "Label.txt"
            write_label_txt(pages, out)
            raw = out.read_text(encoding="utf-8")
            self.assertIn("phân tích", raw)
            # ensure_ascii=False: no unicode escapes
            self.assertNotIn("\\u", raw)

    def test_no_embedded_newlines_in_output(self) -> None:
        pages = [
            PPOCRLabelPage(
                image_path="vi/doc.001.jpg",
                annotations=[
                    PPOCRLabelAnnotation(
                        transcription="line one",
                        points=[[0, 0], [1, 0], [1, 1], [0, 1]],
                    )
                ],
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "Label.txt"
            write_label_txt(pages, out)
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)


# ---------------------------------------------------------------------------
# parse_label_txt — error cases
# ---------------------------------------------------------------------------


class TestParseLabelTxtErrors(unittest.TestCase):
    def _write(self, content: str) -> Path:
        d = tempfile.mkdtemp()
        p = Path(d) / "Label.txt"
        p.write_text(content, encoding="utf-8")
        return p

    def test_bad_json_raises_with_line_number(self) -> None:
        p = self._write("vi/doc.001.jpg\t{not json}\n")
        with self.assertRaises(ValueError) as ctx:
            parse_label_txt(p)
        self.assertIn("1", str(ctx.exception))

    def test_wrong_point_count_raises(self) -> None:
        ann = json.dumps([{"transcription": "x", "points": [[0, 0], [1, 0], [1, 1]]}])
        p = self._write(f"vi/doc.001.jpg\t{ann}\n")
        with self.assertRaises(ValueError) as ctx:
            parse_label_txt(p)
        self.assertIn("1", str(ctx.exception))

    def test_non_integer_coords_raises(self) -> None:
        ann = json.dumps([
            {"transcription": "x", "points": [[0.5, 0], [1, 0], [1, 1], [0, 1]]}
        ])
        p = self._write(f"vi/doc.001.jpg\t{ann}\n")
        with self.assertRaises(ValueError) as ctx:
            parse_label_txt(p)
        self.assertIn("1", str(ctx.exception))

    def test_empty_transcription_raises(self) -> None:
        ann = json.dumps([{"transcription": "", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}])
        p = self._write(f"vi/doc.001.jpg\t{ann}\n")
        with self.assertRaises(ValueError) as ctx:
            parse_label_txt(p)
        self.assertIn("1", str(ctx.exception))

    def test_missing_tab_raises(self) -> None:
        p = self._write("vi/doc.001.jpg no_tab_here\n")
        with self.assertRaises(ValueError) as ctx:
            parse_label_txt(p)
        self.assertIn("1", str(ctx.exception))

    def test_file_not_found_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            parse_label_txt(Path("/nonexistent/Label.txt"))

    def test_line_number_on_second_bad_line(self) -> None:
        good = json.dumps([{"transcription": "ok", "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}])
        bad = json.dumps([{"transcription": "x", "points": [[0, 0], [1, 0], [1, 1]]}])
        p = self._write(f"vi/doc.001.jpg\t{good}\nvi/doc.002.jpg\t{bad}\n")
        with self.assertRaises(ValueError) as ctx:
            parse_label_txt(p)
        self.assertIn("2", str(ctx.exception))


# ---------------------------------------------------------------------------
# write_file_state_txt
# ---------------------------------------------------------------------------


class TestWriteFileStateTxt(unittest.TestCase):
    def _pages(self) -> list[PPOCRLabelPage]:
        return [
            PPOCRLabelPage(
                image_path="vi/doc.001.jpg",
                annotations=[],
            ),
            PPOCRLabelPage(
                image_path="vi/doc.002.jpg",
                annotations=[],
            ),
        ]

    def test_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "fileState.txt"
            write_file_state_txt(self._pages(), out)
            lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "vi/doc.001.jpg\t1")
        self.assertEqual(lines[1], "vi/doc.002.jpg\t1")

    def test_ordering_matches_input(self) -> None:
        pages = [
            PPOCRLabelPage(image_path="vi/doc.003.jpg", annotations=[]),
            PPOCRLabelPage(image_path="vi/doc.001.jpg", annotations=[]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "fileState.txt"
            write_file_state_txt(pages, out)
            lines = out.read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines[0].startswith("vi/doc.003.jpg"))
        self.assertTrue(lines[1].startswith("vi/doc.001.jpg"))

    def test_overwrite_raises_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "fileState.txt"
            write_file_state_txt(self._pages(), out)
            with self.assertRaises(FileExistsError):
                write_file_state_txt(self._pages(), out)

    def test_overwrite_true_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "fileState.txt"
            write_file_state_txt(self._pages(), out)
            write_file_state_txt(self._pages(), out, overwrite=True)


# ---------------------------------------------------------------------------
# migrate_ocr_document_json
# ---------------------------------------------------------------------------


class TestMigrateOcrDocumentJson(unittest.TestCase):
    def _write_legacy(self, pages: list[dict]) -> Path:
        d = tempfile.mkdtemp()
        p = Path(d) / "old_doc.json"
        legacy = {
            "source": "test.pdf",
            "language": "vi",
            "page_count": len(pages),
            "width": 100,
            "height": 200,
            "pages": pages,
        }
        p.write_text(json.dumps(legacy), encoding="utf-8")
        return p

    def _page_dict(self, doc_page: int = 1) -> dict:
        return {
            "pdf_page_number": doc_page,
            "document_page_number": doc_page,
            "page_name": f"{doc_page:03d}",
            "method": "tesseract",
            "paragraphs": [],
        }

    def test_migration_produces_list(self) -> None:
        path = self._write_legacy([self._page_dict(1), self._page_dict(2)])
        migrate_ocr_document_json(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_already_new_format_unchanged(self) -> None:
        d = tempfile.mkdtemp()
        p = Path(d) / "pages.json"
        pages = [self._page_dict(1)]
        p.write_text(json.dumps(pages), encoding="utf-8")
        result_path = migrate_ocr_document_json(p)
        # Must return source path and content must still be a list
        self.assertEqual(result_path, p)
        self.assertIsInstance(json.loads(p.read_text(encoding="utf-8")), list)

    def test_dest_path_used_when_provided(self) -> None:
        path = self._write_legacy([self._page_dict(1)])
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "migrated.json"
            migrate_ocr_document_json(path, dest)
            result = json.loads(dest.read_text(encoding="utf-8"))
            self.assertIsInstance(result, list)

    def test_bad_format_raises(self) -> None:
        d = tempfile.mkdtemp()
        p = Path(d) / "bad.json"
        p.write_text(json.dumps({"no_pages_key": True}), encoding="utf-8")
        with self.assertRaises(ValueError):
            migrate_ocr_document_json(p)

    def test_file_not_found_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            migrate_ocr_document_json(Path("/nonexistent/file.json"))

    def test_pages_loadable_as_ocr_pages(self) -> None:
        path = self._write_legacy([self._page_dict(1), self._page_dict(2)])
        migrate_ocr_document_json(path)
        pages_data = json.loads(path.read_text(encoding="utf-8"))
        from src.schemas import OCRPage
        pages = [OCRPage(**p) for p in pages_data]
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0].document_page_number, 1)


class TestHeaderParsingAndPPOCRExport(unittest.TestCase):
    def test_parse_header_vietnamese_full(self) -> None:
        doc_p, align_p = parse_header_paragraph("- 19 - (  459 )", language="vi")
        self.assertEqual(doc_p, 19)
        self.assertEqual(align_p, 459)

    def test_parse_header_vietnamese_no_align(self) -> None:
        doc_p, align_p = parse_header_paragraph("- 19 -", language="vi")
        self.assertEqual(doc_p, 19)
        self.assertIsNone(align_p)

    def test_parse_header_sinonom_align_none(self) -> None:
        # For sinonom ('ch'), align_to must be None even if secondary numbers exist
        doc_p, align_p = parse_header_paragraph("- 19 - (  459 )", language="ch")
        self.assertEqual(doc_p, 19)
        self.assertIsNone(align_p)

    def test_parse_header_no_numbers(self) -> None:
        doc_p, align_p = parse_header_paragraph("CHƯƠNG THỨ NHẤT", language="vi")
        self.assertIsNone(doc_p)
        self.assertIsNone(align_p)

    def test_ocr_page_to_ppocr_skip_first_paragraph(self) -> None:
        page = _make_page([
            _make_para("- 19 - (  459 )", bbox=(323, 88, 636, 110), pid=1),
            _make_para("TỰA", bbox=(302, 159, 406, 194), pid=2),
            _make_para("Nội dung bài viết", bbox=(60, 242, 640, 387), pid=3),
        ], doc_page=19)
        res = ocr_page_to_ppocr(
            page,
            "vi/doc.019.jpg",
            document_page_number=19,
            align_to=459,
            skip_first_paragraph=True,
        )
        self.assertEqual(len(res.annotations), 2)
        self.assertEqual(res.annotations[0].transcription, "TỰA")
        self.assertEqual(res.annotations[0].document_page_number, 19)
        self.assertEqual(res.annotations[0].align_to, 459)
        self.assertEqual(res.annotations[0].points, [[302, 159], [406, 159], [406, 194], [302, 194]])
        self.assertEqual(res.annotations[1].transcription, "Nội dung bài viết")
        self.assertEqual(res.annotations[1].document_page_number, 19)
        self.assertEqual(res.annotations[1].align_to, 459)

    def test_round_trip_document_page_number(self) -> None:
        pages = [
            PPOCRLabelPage(
                image_path="vi/doc.019.jpg",
                annotations=[
                    PPOCRLabelAnnotation(
                        transcription="TỰA",
                        points=[[1, 2], [3, 2], [3, 4], [1, 4]],
                        document_page_number=19,
                        align_to=459,
                    )
                ],
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "Label.txt"
            write_label_txt(pages, out)
            parsed = parse_label_txt(out)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].annotations[0].document_page_number, 19)
        self.assertEqual(parsed[0].annotations[0].align_to, 459)

    def test_normalize_transcription_punctuation_and_nfc(self) -> None:
        raw = "Giáng sinh , Chúa : thương xót ( chúng ta ) ! và cứu chuộc ."
        expected = "Giáng sinh, Chúa: thương xót (chúng ta)! và cứu chuộc."
        self.assertEqual(normalize_transcription(raw), expected)

    def test_annotation_validation_normalizes_punctuation(self) -> None:
        ann = PPOCRLabelAnnotation(
            transcription="Giáng sinh , Chúa",
            points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        )
        self.assertEqual(ann.transcription, "Giáng sinh, Chúa")


if __name__ == "__main__":
    unittest.main()
