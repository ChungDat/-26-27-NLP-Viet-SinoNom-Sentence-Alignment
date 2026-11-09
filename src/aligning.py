from os import PathLike
from pathlib import Path

import pandas as pd


def align(
    source_1: str | PathLike[str],
    source_2: str | PathLike[str],
    header_1: str,
    header_2: str,
    xlsx_output: str | PathLike[str] = "aligned.xlsx",
    tsv_output: str | PathLike[str] = "aligned.tsv",
) -> tuple[Path, Path]:
    """Align two newline-separated text files and save the result.

    Each input line becomes one row. The two files must contain the same
    number of lines; otherwise a ``ValueError`` is raised to avoid silently
    creating incorrect alignments.

    Args:
        source_1: Path to the first text file.
        source_2: Path to the second text file.
        header_1: Column name for lines from ``source_1``.
        header_2: Column name for lines from ``source_2``.
        xlsx_output: Destination path for the Excel workbook.
        tsv_output: Destination path for the tab-separated file.

    Returns:
        The Excel and TSV output paths, in that order.
    """
    source_1_path = Path(source_1)
    source_2_path = Path(source_2)

    with source_1_path.open("r", encoding="utf-8") as file_1:
        lines_1 = [line.rstrip("\r\n") for line in file_1]
    with source_2_path.open("r", encoding="utf-8") as file_2:
        lines_2 = [line.rstrip("\r\n") for line in file_2]

    if len(lines_1) != len(lines_2):
        raise ValueError(
            f"Input files have different line counts: "
            f"{source_1_path} has {len(lines_1)} lines, "
            f"{source_2_path} has {len(lines_2)} lines."
        )

    aligned = pd.DataFrame({header_1: lines_1, header_2: lines_2})
    xlsx_path = Path(xlsx_output)
    tsv_path = Path(tsv_output)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    aligned.to_excel(xlsx_path, index=False)
    aligned.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")

    return xlsx_path, tsv_path
