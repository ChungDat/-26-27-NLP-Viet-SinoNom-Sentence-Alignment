from __future__ import annotations

from pydantic import BaseModel, Field


class OCRWord(BaseModel):
    bbox: tuple[int, int, int, int] | None = None  # (x_min, y_min, x_max, y_max)
    text: str
    confidence: float | None = None


class OCRLine(BaseModel):
    line_id: int | None = None
    bbox: tuple[int, int, int, int] | None = None  # (x_min, y_min, x_max, y_max)
    text: str
    words: list[OCRWord] = Field(default_factory=list)
    confidence: float | None = None


class OCRParagraph(BaseModel):
    paragraph_id: int
    bbox: tuple[int, int, int, int] | None = None  # (x_min, y_min, x_max, y_max)
    text: str
    words: list[OCRWord] = Field(default_factory=list)


class OCRPage(BaseModel):
    pdf_page_number: int  # page number on pdf reader
    document_page_number: int  # actual page number on document
    page_name: str
    method: str
    paragraphs: list[OCRParagraph] = Field(default_factory=list)


class OCRDocument(BaseModel):
    source: str
    language: str
    page_count: int
    width: int | None = None
    height: int | None = None
    pages: list[OCRPage] = Field(default_factory=list)


__all__ = [
    "OCRWord",
    "OCRLine",
    "OCRParagraph",
    "OCRPage",
    "OCRDocument",
]
