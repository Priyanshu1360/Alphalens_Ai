import re
from pathlib import Path

import pdfplumber

from src.utils.config import Config


YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
QUARTER_PATTERN = re.compile(r"\bq([1-4])\b", re.IGNORECASE)


def extract_metadata(file_name):
    lower_name = file_name.lower()

    report_type = None
    if "10-k" in lower_name:
        report_type = "10-k"
    elif "10-q" in lower_name:
        report_type = "10-q"
    elif "8-k" in lower_name:
        report_type = "8-k"

    year_match = YEAR_PATTERN.search(lower_name)
    quarter_match = QUARTER_PATTERN.search(lower_name)

    return {
        "report_type": report_type,
        "year": year_match.group(1) if year_match else None,
        "quarter": quarter_match.group(1) if quarter_match else None,
    }


def _clean_cell(cell):
    if cell is None:
        return ""
    return " ".join(str(cell).split())


def _table_to_text(table):
    rows = []
    for row in table:
        cleaned_row = [_clean_cell(cell) for cell in row]
        if any(cleaned_row):
            rows.append(" | ".join(cleaned_row))
    return "\n".join(rows)


def _extract_page_content(page):
    text_parts = []
    page_text = page.extract_text() or ""
    if page_text.strip():
        text_parts.append(page_text.strip())

    raw_tables = page.extract_tables() or []
    formatted_tables = []
    for table in raw_tables:
        table_text = _table_to_text(table)
        if table_text:
            formatted_tables.append(table_text)

    if formatted_tables:
        text_parts.append("\n\n".join(formatted_tables))

    return "\n\n".join(text_parts), formatted_tables


def load_pdfs(base_path=None):
    documents = []
    root = Path(base_path or Config.PDF_BASE_PATH)

    if not root.exists():
        raise FileNotFoundError(f"PDF directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root}")

    for company_dir in sorted(root.iterdir()):
        if not company_dir.is_dir():
            continue

        for pdf_path in sorted(company_dir.glob(Config.PDF_GLOB_PATTERN)):
            text_parts = []
            tables = []

            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text, page_tables = _extract_page_content(page)
                    if page_text:
                        text_parts.append(page_text)
                    if page_tables:
                        tables.extend(page_tables)

            metadata = extract_metadata(pdf_path.name)

            documents.append(
                {
                    "text": "\n\n".join(text_parts).strip(),
                    "tables": tables,
                    "company": company_dir.name,
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path),
                    "report_type": metadata["report_type"],
                    "year": metadata["year"],
                    "quarter": metadata["quarter"],
                }
            )

    return documents
