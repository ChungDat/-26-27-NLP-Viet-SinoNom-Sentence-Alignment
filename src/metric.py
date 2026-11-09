from itertools import zip_longest
from os import PathLike
from pathlib import Path


def _read_paragraphs(path: str | PathLike[str]) -> list[str]:
    """
    Read one paragraph per line, without the line-ending characters
    """
    return Path(path).read_text(encoding="utf-8").splitlines()


def _edit_distance(first: list[str], second: list[str]) -> int:
    """
    Return the Levenshtein distance between two sequences
    """
    previous = list(range(len(second) + 1))
    for first_index, first_item in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_item in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,  # insertion
                    previous[second_index] + 1,  # deletion
                    previous[second_index - 1] + (first_item != second_item),
                )
            )
        previous = current
    return previous[-1]


def _paragraph_distances(
    hypothesis: list[str], reference: list[str], *, words: bool
) -> tuple[int, int]:
    """
    Return total errors and total reference units for aligned paragraphs
    """
    errors = 0
    reference_units = 0
    for hypothesis_paragraph, reference_paragraph in zip_longest(
        hypothesis, reference, fillvalue=""
    ):
        hypothesis_units = (
            hypothesis_paragraph.split()
            if words
            else list(hypothesis_paragraph)
        )
        reference_units_for_paragraph = (
            reference_paragraph.split()
            if words
            else list(reference_paragraph)
        )
        errors += _edit_distance(hypothesis_units, reference_units_for_paragraph)
        reference_units += len(reference_units_for_paragraph)
    return errors, reference_units


def _error_rate(errors: int, reference_units: int) -> float:
    """
    Convert counts to a rate, handling empty reference files explicitly
    """
    if reference_units == 0:
        return 0.0 if errors == 0 else float("inf")
    return errors / reference_units


def CER(
    hypothesis_path: str | PathLike[str], reference_path: str | PathLike[str]
) -> float:
    """
    Return the character error rate for two paragraph-aligned text files

    Character errors are summed over all lines and divided by the total number
    of characters in the reference file. Spaces are treated as characters;
    line-ending characters are not included
    """
    errors, reference_characters = _paragraph_distances(
        _read_paragraphs(hypothesis_path),
        _read_paragraphs(reference_path),
        words=False,
    )
    return _error_rate(errors, reference_characters)


def WER(
    hypothesis_path: str | PathLike[str], reference_path: str | PathLike[str]
) -> float:
    """
    Return the word error rate for two paragraph-aligned text files

    Words are whitespace-delimited, and word errors are divided by the total
    number of words in the reference file
    """
    errors, reference_words = _paragraph_distances(
        _read_paragraphs(hypothesis_path),
        _read_paragraphs(reference_path),
        words=True,
    )
    return _error_rate(errors, reference_words)
