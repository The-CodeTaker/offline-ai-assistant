"""
utils/file_parser.py — Universal document text extractor.

Supported formats
-----------------
  .txt              Native Python open()
  .pdf              PyMuPDF (fitz)                    pip install pymupdf
  .csv              pandas                             pip install pandas
  .xlsx / .xls      pandas + openpyxl                  pip install pandas openpyxl
  .png / .jpg / .jpeg / .bmp / .tiff / .webp
                    Ollama llava vision model (primary)
                    pytesseract + Pillow (fallback OCR) pip install pytesseract pillow

Image analysis strategy
-----------------------
  1. PRIMARY — Ollama llava:
       Sends the raw image bytes to a locally-running llava model via the
       Ollama Python client.  The prompt asks the model to describe the image
       in detail AND extract any visible text, giving the RAG pipeline rich
       semantic content that OCR alone cannot produce.
  2. FALLBACK — pytesseract:
       Used when llava is not available (model not pulled, Ollama unreachable,
       or any runtime error).  Returns raw OCR text, which is better than
       nothing for document images.

Design principles
-----------------
- Every format handler is in its own try/except — a missing dependency or
  corrupt file never propagates an exception to the caller.
- parse_file() always returns str (empty string on failure).
"""

from __future__ import annotations

import base64
import os
from loguru import logger

# Vision model used for image analysis — pull with:  ollama pull llava
_VISION_MODEL = "llava-llama3"

# Prompt sent to the vision model
_VISION_PROMPT = (
    "Please do two things:\n"
    "1. Describe this image in comprehensive detail — include objects, people, "
    "colours, layout, and any contextual information you can infer.\n"
    "2. Transcribe ALL text visible in the image exactly as written.\n\n"
    "Format your response as:\n"
    "DESCRIPTION: <your detailed description>\n"
    "EXTRACTED TEXT: <verbatim text from the image, or 'None' if absent>"
)


class FileParser:
    """
    Stateless utility class for extracting text from common document types.

    Usage:
        text = FileParser.parse_file("/path/to/report.pdf")
        text = FileParser.parse_file("/path/to/chart.png")
    """

    @staticmethod
    def parse_file(filepath: str) -> str:
        """
        Extract and return plain text from *filepath*.

        Returns "" on any error so callers never receive an exception.
        The returned string is what gets embedded into the vector store,
        so richer descriptions produce better retrieval quality.
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
            logger.debug(
                f"FileParser: read PDF '{filepath}' "
                f"({len(text)} chars, {len(pages)} pages)"
            )
            return text
        except Exception as exc:
            logger.error(f"FileParser._read_pdf: {exc}")
            return ""

    @staticmethod
    def _read_csv(filepath: str) -> str:
        """
        Load a CSV with pandas and convert to a Markdown table.
        Falls back to a plain string if tabulate is absent.
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            logger.warning("FileParser: pandas not installed. pip install pandas")
            return ""

        try:
            df = pd.read_csv(filepath)
            try:
                text = df.to_markdown(index=False)
            except Exception:
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
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            logger.warning("FileParser: pandas not installed. pip install pandas openpyxl")
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
        Analyse an image file and return a rich description + any extracted text.

        Strategy
        --------
        PRIMARY — Ollama llava vision model:
            Encodes the image as base64 and sends it to the locally-running
            llava model.  The response includes a comprehensive description
            of the image contents AND a verbatim transcription of any visible
            text.  This gives the RAG pipeline genuine semantic understanding
            rather than just character strings from OCR.

            Requires:  ollama pull llava   (one-time download, ~4 GB)

        FALLBACK — pytesseract OCR:
            If llava is unavailable for any reason (model not pulled, Ollama
            not running, network timeout), we fall back to pytesseract OCR.
            This still extracts printed text but has no scene understanding.

            Requires:  pip install pytesseract pillow
                       + Tesseract binary on OS PATH

        Returns "" only if both strategies fail.
        """
        # ── Strategy 1: Ollama llava vision model ─────────────────────────
        try:
            import ollama as _ollama

            # Read and base64-encode the image bytes
            with open(filepath, "rb") as img_file:
                image_bytes  = img_file.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            logger.info(
                f"FileParser: sending '{os.path.basename(filepath)}' "
                f"to {_VISION_MODEL} for vision analysis..."
            )

            response = _ollama.chat(
                model=_VISION_MODEL,
                messages=[
                    {
                        "role":    "user",
                        "content": _VISION_PROMPT,
                        "images":  [image_b64],
                    }
                ],
            )
            description = response.message.content.strip()

            if description:
                logger.info(
                    f"FileParser: llava described '{os.path.basename(filepath)}' "
                    f"({len(description)} chars)"
                )
                # Prepend a source label so the LLM knows this is image content
                return f"[Image analysis of '{os.path.basename(filepath)}']\n\n{description}"

            logger.warning("FileParser: llava returned an empty description — falling back to OCR")

        except Exception as exc:
            # Common causes: model not pulled, Ollama not running, import missing
            logger.warning(
                f"FileParser: llava vision failed ({exc!r}) — "
                "falling back to pytesseract OCR. "
                f"To enable vision analysis: ollama pull {_VISION_MODEL}"
            )

        # ── Strategy 2: pytesseract OCR fallback ──────────────────────────
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError:
            logger.warning(
                "FileParser: pytesseract / Pillow not installed and llava is unavailable. "
                "Image analysis is completely disabled for this file.\n"
                "  Enable vision:  ollama pull llava\n"
                "  Enable OCR:     pip install pytesseract pillow"
            )
            return ""

        try:
            img  = Image.open(filepath)
            text = pytesseract.image_to_string(img).strip()
            logger.debug(
                f"FileParser: OCR'd '{os.path.basename(filepath)}' "
                f"({len(text)} chars)"
            )
            if not text:
                return (
                    f"[Image '{os.path.basename(filepath)}' — "
                    "no text detected by OCR and vision model was unavailable]"
                )
            return (
                f"[OCR text from '{os.path.basename(filepath)}']\n\n{text}"
            )
        except Exception as exc:
            logger.error(f"FileParser._read_image OCR fallback failed: {exc}")
            return ""
