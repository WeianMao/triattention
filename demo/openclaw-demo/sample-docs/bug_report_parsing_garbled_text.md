# Bug Report: Garbled Chinese After PDF Parsing

**Reporter**: Liu Yang | **Date**: 2026-02-20 | **Priority**: P1 | **Status**: Interim fix deployed

---

## 1. Issue

Three customer PDFs (127 pages total) produced ~30% garbled Chinese text after doc-parser processing. English and numeric content remained correct. Symptoms: Unicode replacement characters (U+FFFD) or incorrect CJK glyphs.

## 2. Impact

- **Direct** - garbled chunks cannot be indexed, so semantic search fully misses them.
- **Indirect** - damages user trust in system accuracy.
- **Scope** - PDFs exported from WPS Office that embed Founder fonts (~15-20% of the client's repository).

## 3. Reproduction

| Condition | Details |
|-----------|---------|
| Source | WPS Office 2023 exports |
| Fonts | Founder Xiaobiaosong / FangSong (embedded) |
| Encoding | CID-based (no ToUnicode mapping) |
| Parser | pdfplumber 0.11.x (pdfminer backend) |

Steps:
1. Create a WPS doc using Founder fonts.
2. Export to PDF.
3. Upload to DocMind.
4. Parsed output shows garbled placeholders ("???") or incorrect glyphs.

## 4. Root Cause

pdfplumber relies on pdfminer for PDF streams. pdfminer needs a ToUnicode CMap to map glyphs. In these PDFs:

1. WPS embeds Adobe-GB1 CID fonts without a complete ToUnicode table.
2. pdfminer falls back to raw CID codes instead of valid Unicode.
3. ~30% of Chinese characters fail to decode.

PyMuPDF (MuPDF) uses richer CID fallback tables, so it handles most of the same files correctly.

## 5. Fixes

### 5.1 Interim Fix (Done)

Added fallback logic:

```python
def extract_text(pdf_path: str) -> list[Section]:
    sections = pdfplumber_extract(pdf_path)
    garbled_ratio = count_garbled_chars(sections) / total_chars(sections)

    if garbled_ratio > 0.10:
        logger.warning("High garble ratio %.1f%%, falling back to PyMuPDF", garbled_ratio * 100)
        fitz_sections = pymupdf_extract(pdf_path)
        sections = merge_extractions(sections, fitz_sections)

    return sections
```

For any page with >10% garbled rate we replace pdfplumber text with PyMuPDF output. Tables still come from pdfplumber (better detection).

**Result**: garble rate dropped from 30% to 5%. Remaining errors are rare fonts or handwritten notes.

### 5.2 Root Fix (Planned)

Trigger OCR when quality is still poor:

1. After fallback, if garbled ratio >5%, run PaddleOCR on that page.
2. Compare OCR vs. text extraction by confidence and pick the best.
3. Annotate each paragraph with its source (text_extract/ocr) and confidence.

Effort: 3 person-days. Overhead: OCR adds 2-3 s/page but only affects ~15% of PDFs.

### 5.3 Long-Term

1. Contribute CID fallback mappings to pdfminer (issue filed by Zhao Lei).
2. Build a font compatibility test suite covering major Chinese fonts.
3. Consider Adobe PDF Services API as a premium fallback (for VIP customers).

## 6. Verification

| File | Pages | Garble (pre) | Garble (post) | Notes |
|------|-------|--------------|---------------|-------|
| Contract_ACME_2026.pdf | 42 | 35% | 3% | pdfplumber + fitz |
| TechSpec_V2.pdf | 85 | 28% | 6% | pdfplumber + fitz |
| Procurement_Q4_2025.pdf | 12 | 31% | 4% | pdfplumber + fitz |

## 7. Status

- [Done] Interim fix merged to `dev` (commit a3f2e1d)
- [Warning] OCR fallback pending PM prioritization
- [Warning] pdfminer issue open, awaiting response
- [Note] Wang Hao wants the root fix before M2 to avoid quality issues during trials

---

**CC**: Wang Hao, Zhao Lei, Zhang Ming
