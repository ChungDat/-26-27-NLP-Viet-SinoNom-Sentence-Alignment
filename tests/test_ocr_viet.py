from pathlib import Path
import unittest

# Test text logic independently from OCR engine execution.
from src.ocr_viet import filtered_ocr_text, normalize_text

try:
    import numpy as np
    from PIL import Image

    from src.ocr_viet import (
        OCRDocument,
        OCRPage,
        VLMOCR,
        TesseractOCR,
        to_image_bytes,
        to_pil_image,
    )

    CV2_PIL_AVAILABLE = True
except ImportError:
    CV2_PIL_AVAILABLE = False


class TestOCRViet(unittest.TestCase):
    def test_filter_bounding_boxes_negative_dimensions(self):
        sample_data = {"text": ["a"], "width": [10], "height": [10]}
        ocr = TesseractOCR(lang="vie")
        with self.assertRaises(ValueError):
            ocr.filter_bounding_boxes(sample_data, min_width=-1, min_height=10)
        with self.assertRaises(ValueError):
            ocr.filter_bounding_boxes(sample_data, min_width=10, min_height=-1)

    def test_filter_bounding_boxes_filtering(self):
        data = {
            "text": ["valid", "too_narrow", "too_short", "good"],
            "left": [10, 20, 30, 40],
            "top": [10, 20, 30, 40],
            "width": [50, 5, 50, 60],
            "height": [20, 20, 4, 30],
            "conf": [95.0, 90.0, 80.0, 99.0],
            "non_list": "metadata",
        }
        ocr = TesseractOCR(lang="vie")
        filtered = ocr.filter_bounding_boxes(data, min_width=10, min_height=10)
        self.assertEqual(filtered["text"], ["valid", "good"])
        self.assertEqual(filtered["left"], [10, 40])
        self.assertEqual(filtered["width"], [50, 60])
        self.assertEqual(filtered["height"], [20, 30])
        self.assertEqual(filtered["conf"], [95.0, 99.0])
        self.assertEqual(filtered["non_list"], "metadata")

    def test_filtered_ocr_text(self):
        data = {
            "text": ["Xin", "chào", "Việt", "Nam.", "Thêm", "đoạn."],
            "conf": [90.0, 95.0, 92.0, 98.0, 89.0, 96.0],
            "block_num": [1, 1, 1, 1, 1, 1],
            "par_num": [1, 1, 1, 1, 2, 2],
            "line_num": [1, 1, 2, 2, 1, 2],
        }
        text = filtered_ocr_text(data)
        self.assertEqual(text, "Xin chào Việt Nam.\n\nThêm đoạn.\n")

    def test_tesseract_page_uses_flat_paragraph_words(self):
        ocr = TesseractOCR(lang="vie")
        raw = {
            "text": ["Xin", "chào", "Việt", "Nam."],
            "conf": [90.0, 95.0, 92.0, 98.0],
            "left": [10, 40, 10, 40],
            "top": [10, 10, 40, 40],
            "width": [25, 30, 25, 30],
            "height": [10, 10, 10, 10],
            "block_num": [1, 1, 1, 1],
            "par_num": [1, 1, 1, 1],
            "line_num": [1, 1, 2, 2],
        }

        page = ocr._build_page(
            raw, pdf_page_number=1, document_page_number=1, page_name="test"
        )
        paragraph = page.paragraphs[0]
        self.assertEqual(paragraph.text, "Xin chào Việt Nam.")
        self.assertEqual(
            [word.text for word in paragraph.words],
            ["Xin", "chào", "Việt", "Nam."],
        )
        self.assertIn("Xin chào Việt Nam.", ocr._page_to_text(page))

        document = OCRDocument(
            source="test",
            language="vie",
            page_count=1,
            pages=[page],
        )
        serialized_paragraph = document.model_dump()["pages"][0]["paragraphs"][0]
        self.assertNotIn("lines", serialized_paragraph)
        self.assertEqual(
            [word["text"] for word in serialized_paragraph["words"]],
            ["Xin", "chào", "Việt", "Nam."],
        )

    def test_tesseract_merges_close_paragraphs_and_keeps_word_order(self):
        ocr = TesseractOCR(lang="vie")
        raw = {
            "text": ["đoạn", "tiếp."],
            "conf": [90.0, 95.0],
            "left": [10, 10],
            "top": [10, 23],
            "width": [30, 40],
            "height": [10, 10],
            "block_num": [1, 1],
            "par_num": [1, 2],
            "line_num": [1, 1],
        }

        page = ocr._build_page(
            raw, pdf_page_number=1, document_page_number=1, page_name="test"
        )
        self.assertEqual(len(page.paragraphs), 1)
        self.assertEqual(page.paragraphs[0].text, "đoạn tiếp.")
        self.assertEqual(
            [word.text for word in page.paragraphs[0].words],
            ["đoạn", "tiếp."],
        )

    def test_tesseract_merges_close_blocks_before_paragraphs(self):
        ocr = TesseractOCR(lang="vie")
        raw = {
            "text": ["gần", "cũng-gần", "xa"],
            "conf": [90.0, 95.0, 91.0],
            "left": [10, 10, 10],
            "top": [10, 24, 40],
            "width": [30, 40, 25],
            "height": [10, 10, 10],
            "block_num": [1, 2, 3],
            "par_num": [1, 1, 1],
            "line_num": [1, 1, 1],
        }

        page = ocr._build_page(
            raw, pdf_page_number=1, document_page_number=1, page_name="test"
        )
        self.assertEqual(
            [paragraph.text for paragraph in page.paragraphs],
            ["gần cũng-gần", "xa"],
        )

    def test_sample_tesseract_text_matches_corrected_output(self):
        fixture_dir = Path(__file__).resolve().parents[1] / "Dong_hang" / "test"
        page = OCRPage.model_validate_json(
            (fixture_dir / "hypo_tesseract.json").read_text(encoding="utf-8")
        )
        actual = normalize_text(TesseractOCR._page_to_text(page))
        expected = (fixture_dir / "hypo_tesseract_correct.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(actual, expected)

    def test_normalize_text(self):
        raw = "Đây là dòng 1\nvà dòng 2 .  Tiếp theo.\n"
        norm = normalize_text(raw)
        self.assertEqual(norm, "Đây là dòng 1 và dòng 2. Tiếp theo.\n")

    def test_to_pil_image_and_to_image_bytes(self):
        if not CV2_PIL_AVAILABLE:
            self.skipTest("cv2/numpy not installed in current test environment")
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        pil_img = to_pil_image(arr)
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.size, (100, 100))

        png_bytes = to_image_bytes(arr)
        self.assertIsInstance(png_bytes, bytes)
        self.assertTrue(len(png_bytes) > 0)
        self.assertEqual(png_bytes[:8], b"\x89PNG\r\n\x1a\n")

    def test_annotate_image(self):
        if not CV2_PIL_AVAILABLE:
            self.skipTest("cv2/numpy not installed in current test environment")
        arr = np.zeros((100, 200, 3), dtype=np.uint8)
        data = {
            "text": ["Test", "Low"],
            "conf": [95.0, 50.0],
            "left": [10, 60],
            "top": [20, 20],
            "width": [40, 40],
            "height": [20, 20],
        }
        ocr = TesseractOCR(lang="vie")
        ocr._image = arr
        ocr._raw = data
        ocr._filtered_raw = data
        ocr._page = OCRPage(
            pdf_page_number=1,
            document_page_number=1,
            page_name="test",
            method="tesseract",
        )
        annotated = ocr.annotate_image()
        self.assertIsInstance(annotated, Image.Image)
        self.assertEqual(annotated.size, (200, 100))

    def test_tesseract_ocr_require_run(self):
        ocr = TesseractOCR(lang="vie")
        with self.assertRaises(RuntimeError):
            _ = ocr.page
        with self.assertRaises(RuntimeError):
            _ = ocr.text
        with self.assertRaises(RuntimeError):
            _ = ocr.annotate_image()

    def test_vlm_ocr_require_run(self):
        ocr = VLMOCR()
        with self.assertRaises(RuntimeError):
            _ = ocr.page
        with self.assertRaises(RuntimeError):
            _ = ocr.text

    def test_tesseract_ocr_mocked_run(self):
        ocr = TesseractOCR(lang="vie")
        fake_raw = {
            "text": ["Hà", "Nội."],
            "conf": [95.0, 92.0],
            "left": [10, 50],
            "top": [10, 10],
            "width": [30, 40],
            "height": [20, 20],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 1],
        }

        # Inject fake data into _build_page
        ocr._image = Image.new("RGB", (200, 100), color="white")
        ocr._raw = fake_raw
        ocr._filtered_raw = fake_raw
        ocr._page = ocr._build_page(
            fake_raw, pdf_page_number=1, document_page_number=1, page_name="test"
        )

        self.assertIn("Hà Nội.", ocr.text)
        self.assertEqual(ocr.text, "Hà Nội.\n")
        annotated = ocr.annotate_image()
        self.assertIsInstance(annotated, Image.Image)

    def test_vlm_ocr_mocked_run(self):
        ocr = VLMOCR()
        raw_text = "Hà Nội là thủ đô.\n\nViệt Nam tươi đẹp."
        ocr._raw_text = raw_text
        ocr._page = ocr._build_page(
            normalized_text=normalize_text(raw_text),
            pdf_page_number=1,
            document_page_number=1,
            page_name="test",
        )
        self.assertEqual(
            normalize_text(raw_text),
            "Hà Nội là thủ đô.\n\nViệt Nam tươi đẹp.\n",
        )
        self.assertIn("Hà Nội là thủ đô.", ocr.text)
        self.assertIn("Việt Nam tươi đẹp.", ocr.text)
        self.assertEqual(
            ocr.text,
            "Hà Nội là thủ đô.\n\nViệt Nam tươi đẹp.\n",
        )
        self.assertEqual(
            ocr.page.paragraphs[0].text,
            "Hà Nội là thủ đô.",
        )
        self.assertEqual(
            [word.text for word in ocr.page.paragraphs[0].words],
            ["Hà", "Nội", "là", "thủ", "đô."],
        )
        self.assertNotIn("lines", ocr.page.model_dump()["paragraphs"][0])


if __name__ == "__main__":
    unittest.main()
