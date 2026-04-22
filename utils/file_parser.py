"""
utils/file_parser.py — Universal document text extractor.

Supported formats
-----------------
  .txt              Native Python open()
  .pdf              PyMuPDF (fitz)          pip install pymupdf
  .csv              pandas                  pip install pandas
  .xlsx / .xls      pandas + openpyxl/xlrd  pip install pandas openpyxl
  .png / .jpg / .jpeg  pytesseract + Pillow pip install pytesseract pillow
                    (Tesseract OCR must also be installed on the OS)

Design principles
-----------------
- Every format handler is wrapped in its own try/except so a missing
  optional dependency or corrupt file never crashes the caller.
- On any failure the method returns "" so the assistant gracefully
  falls back to answering without document context.
- Conversion to a readable string is format-aware:
    CSV/XLSX → Markdown table (via pandas .to_markdown()) with a plain-text
               fallback if tabulate is not installed.
    Images   → raw OCR text with leading/trailing whitespace stripped.
"""

from __future__ import annotations

import os
from loguru import logger


class FileParser:
    """
    Stateless utility class for extracting text from common document types.

    Usage:
        text = FileParser.parse_file("/path/to/report.pdf")
    """

    @staticmethod
    def parse_file(filepath: str) -> str:
        """
        Extract and return plain text from *filepath*.

        Returns "" on any error so callers never receive an exception.
        """
        if not os.path.isfile(filepath):
            logger.warning(f"FileParser: path does not exist: {filepath!r}")
            return ""

        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext == ".txt":
                return FileParser._read_txt(filepath)
            elif ext == ".pdf":
                return FileParser._read_pdf(filepath)
            elif ext == ".csv":
                return FileParser._read_csv(filepath)
            elif ext in (".xlsx", ".xls"):
                return FileParser._read_excel(filepath)
            elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
                return FileParser._read_image(filepath)
            else:
                # Fallback: try reading as UTF-8 text for unknown types
                logger.debug(f"FileParser: unknown extension {ext!r}, trying plain text")
                return FileParser._read_txt(filepath)
        except Exception as exc:
            logger.error(f"FileParser.parse_file({filepath!r}): {exc}")
            return ""

    # ------------------------------------------------------------------
    # Format handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_txt(filepath: str) -> str:
        """Read a plain-text file, tolerating encoding errors."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            logger.debug(f"FileParser: read TXT '{filepath}' ({len(text)} chars)")
            return text
        except Exception as exc:
            logger.error(f"FileParser._read_txt: {exc}")
            return ""

    @staticmethod
    def _read_pdf(filepath: str) -> str:
        """Extract text from every page of a PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning(
                "FileParser: PyMuPDF not installed. "
                "Install with:  pip install pymupdf"
            )
            return ""

        try:
            doc   = fitz.open(filepath)
            pages = [page.get_text() for page in doc]
            doc.close()
            text = "\n".join(pages)
            logger.debug(f"FileParser: read PDF '{filepath}' ({len(text)} chars, {len(pages)} pages)")
            return text
        except Exception as exc:
            logger.error(f"FileParser._read_pdf: {exc}")
            return ""

    @staticmethod
    def _read_csv(filepath: str) -> str:
        """
        Load a CSV with pandas and convert to a Markdown table.
        Falls back to a comma-separated string if tabulate is absent.
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            logger.warning(
                "FileParser: pandas not installed. "
                "Install with:  pip install pandas"
            )
            return ""

        try:
            df = pd.read_csv(filepath)
            try:
                text = df.to_markdown(index=False)
            except Exception:
                # tabulate not installed — use simple CSV string
                text = df.to_string(index=False)
            logger.debug(f"FileParser: read CSV '{filepath}' ({df.shape[0]} rows)")
            return text or ""
        except Exception as exc:
            logger.error(f"FileParser._read_csv: {exc}")
            return ""

    @staticmethod
    def _read_excel(filepath: str) -> str:
        """
        Load an Excel workbook with pandas.  All sheets are concatenated with
        sheet-name headers so the model knows which sheet data came from.
        Falls back to a tabular string if tabulate is absent.
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            logger.warning(
                "FileParser: pandas not installed. "
                "Install with:  pip install pandas openpyxl"
            )
            return ""

        try:
            sheets = pd.read_excel(filepath, sheet_name=None)
            parts: list[str] = []
            for sheet_name, df in sheets.items():
                parts.append(f"### Sheet: {sheet_name}")
                try:
                    parts.append(df.to_markdown(index=False))
                except Exception:
                    parts.append(df.to_string(index=False))
            text = "\n\n".join(parts)
            logger.debug(f"FileParser: read Excel '{filepath}' ({len(sheets)} sheets)")
            return text
        except Exception as exc:
            logger.error(f"FileParser._read_excel: {exc}")
            return ""

    @staticmethod
    def _read_image(filepath: str) -> str:
        """
        OCR an image file using pytesseract + Pillow.

        Requires:
          pip install pytesseract pillow
          Tesseract OCR binary installed on the OS
            Windows: https://github.com/UB-Mannheim/tesseract/wiki
            Linux:   sudo apt install tesseract-ocr
        """
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError:
            logger.warning(
                "FileParser: pytesseract / Pillow not installed. "
                "Install with:  pip install pytesseract pillow"
            )
            return ""

        try:
            img  = Image.open(filepath)
            text = pytesseract.image_to_string(img).strip()
            logger.debug(f"FileParser: OCR'd image '{filepath}' ({len(text)} chars)")
            return text
        except Exception as exc:
            logger.error(f"FileParser._read_image: {exc}")
            return ""
