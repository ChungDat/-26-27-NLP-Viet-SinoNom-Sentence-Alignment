from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytesseract


def select_pages(start: int, end: int, step: int = 1) -> list[int]:
    """Create a list of selected page numbers.

    Args:
        start: The first page to select.
        end: The last page to select, inclusive.
        step: The number of pages to skip between selections.

    Returns:
        A list of 1-indexed page numbers between ``start`` and ``end``.
    """

    if start <= end:
        pages = list(range(start, end + 1, step))
    else:
        pages = list(range(start, end - 1, -step))
    return pages


def pdf_to_images(
    pdf_path: str,
    pages: list[int],
    dpi: int = 300,
    orientation_threshold: float = 3.0,
) -> list[np.ndarray]:
    """Convert selected PDF pages into numpy ndarray images (in RGB format).

    Args:
        pdf_path: Path to the PDF file.
        pages: List of 1-indexed page numbers to extract.
        dpi: Rendering resolution in dots per inch.
        orientation_threshold: Minimum orientation confidence required to
            process a page.

    Returns:
        A list of extracted images in the requested output mode.
    """

    document = pymupdf.open(pdf_path)
    images = []

    for page_number in pages:
        if not 1 <= page_number <= len(document):
            print(f"Skip invalid page {page_number}")

        page = document[page_number - 1]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)

        image = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        ).reshape(pixmap.height, pixmap.width, pixmap.n)

        orientation_data = pytesseract.image_to_osd(
            image,
            output_type=pytesseract.Output.DICT,
        )

        if orientation_data["orientation_conf"] < orientation_threshold:
            print(
                f"Page {page_number} has orientation confidence lower than "
                f"{orientation_threshold}"
            )
            continue

        if orientation_data["rotate"] == 0:
            pass
        elif orientation_data["rotate"] == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation_data["rotate"] == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif orientation_data["rotate"] == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            print(f"Page {page_number} has unusual orientation")

        images.append(image)

    document.close()
    return images


def save_images(
    destination_path: str,
    images: list[np.ndarray],
    pages: list[int],
) -> None:
    """Save extracted page images to a directory.

    Args:
        destination_path: Path to the destination directory.
        images: Images to save.
        pages: List of 1-indexed page numbers corresponding to ``images``.
    """

    destination = Path(destination_path)
    for image, page_number in zip(images, pages):
        file_name = f"{page_number:04d}.png"
        cv2.imwrite(str(destination / file_name), image)


def main() -> None:
    """Extract sample pages from a PDF in each supported image mode."""

    pages = select_pages(40, 40, 1)
    pdf_path = "CV19-Phạm Phúc An Khang-20120303/T1_Sách Truyện Các Thánh (Tháng 1).pdf"


    destination_path = f"Dong_hang/test/"
    images = pdf_to_images(pdf_path, pages, dpi=250)
    save_images(destination_path, images, pages)


if __name__ == "__main__":
    main()
