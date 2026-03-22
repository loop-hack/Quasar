import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter

import pymupdf


#  Fixing encoding artifacts

def fix_encoding(text: str) -> str:

    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u2018\u2019\u02bc\u0060\u00b4]", "'", text)
    text = re.sub(r"[\u201c\u201d\u00ab\u00bb\u2039\u203a]", '"', text)
    text = re.sub(r"[\u2013\u2014\u2015\u2212\u2010\u2011]", " - ", text)
    text = text.replace("\u2026", "...")
    text = re.sub(r"[\u2022\u2023\u25e6\u2043\u2219]", "-", text)
    text = text.replace("\u00ad", "")
    text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]", "", text)
    text = re.sub(r"[\x00\ufffd]", "", text)
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text


# Remove headers footers and page numbers

def remove_headers_footers(text: str) -> str:

    text = re.sub(
        r"^\s*[-\[\(]?\s*\d{1,4}\s*[-\]\)]?\s*$",
        "", text, flags=re.MULTILINE
    )

    text = re.sub(
        r"\b(page|pg|p\.?)\s*\d+(\s+of\s+\d+)?\b",
        "", text, flags=re.IGNORECASE
    )

    text = re.sub(
        r"^\s*[ivxlcdmIVXLCDM]{1,8}\s*$",
        "", text, flags=re.MULTILINE
    )

    text = re.sub(
        r"^\s*(chapter|section|part|unit|module)\s+[\dIVXivx]+\s*$",
        "", text, flags=re.MULTILINE | re.IGNORECASE
    )

    text = re.sub(
        r"^.*?(copyright|all rights reserved|unauthorized reproduction).*?$",
        "", text, flags=re.MULTILINE | re.IGNORECASE
    )

    text = re.sub(r"^\s*https?://\S+\s*$", "", text, flags=re.MULTILINE)

    text = re.sub(
        r"^\s*(isbn|doi|issn)[:\s][\d\-X\.\/]+\s*$",
        "", text, flags=re.MULTILINE | re.IGNORECASE
    )

    return text


#  Fixing broken line wrapping

def fix_line_breaks(text: str) -> str:

    ends_mid = re.compile(r"[^.?!:;\"']\s*$")

    new_block = re.compile(
        r"^\s*$"                          # blank line
        r"|^\s{4,}"                       # heavily indented (code)
        r"|^\s*\d+[\.\)]\s+"             # numbered list: "1. " or "1) "
        r"|^\s*[-\u2022\u25e6\u2023]\s+" # bullet point
        r"|^\s*[A-Z][A-Z\s]{4,}$"        # ALL CAPS heading
        r"|^\s*#{1,6}\s"                  # markdown heading
    )

    lines = text.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        while (
            i + 1 < len(lines)
            and ends_mid.search(line)
            and not new_block.match(lines[i + 1])
            and len(lines[i + 1].strip()) > 0
        ):
            i += 1
            line = line.rstrip() + " " + lines[i].lstrip()
        result.append(line)
        i += 1

    return "\n".join(result)


#  Removing symbols and noise

def remove_noise(text: str) -> str:

    text = re.sub(r"^[\s\-=_*#~+\.]{4,}\s*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"([^\w\s])\1{2,}", r"\1", text)

    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}(.+?)_{1,2}",   r"\1", text)

    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    text = re.sub(r"<[^>]+>", " ", text)

    html_entities = {
        "&nbsp;": " ", "&amp;": "&", "&lt;": "<",
        "&gt;": ">",   "&quot;": '"', "&apos;": "'",
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    text = re.sub(r"&[a-z]{2,6};", " ", text)

    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)

    text = re.sub(r"https?://\S+", "", text)

    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\|", " ", text)

    return text


#  Normalizing whitespace

def normalize_whitespace(text: str) -> str:

    text = re.sub(r"\t", " ", text)

    text = re.sub(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]", " ", text)

    text = re.sub(r"[^\S\n]+", " ", text)

    text = re.sub(r" +$", "", text, flags=re.MULTILINE)

    text = re.sub(r"^ +", "", text, flags=re.MULTILINE)

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    return text


#  Fixing punctuation

def fix_punctuation(text: str) -> str:

    text = re.sub(r"\s+([,\.!?;:'\)])", r"\1", text)
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

    text = re.sub(r",([^\s\d])", r", \1", text)

    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)

    text = re.sub(r"([!?]){2,}", r"\1", text)

    text = re.sub(r"\.{4,}", "...", text)

    text = re.sub(r"(?<!\w)\bi\b(?!\w)", "I", text)

    def cap_after_period(m):
        return m.group(1) + " " + m.group(2).upper()
    text = re.sub(r"(\.) ([a-z])", cap_after_period, text)

    return text


# Removing boilerplate content

def remove_boilerplate(text: str) -> str:

    text = re.sub(
        r"^.{3,60}\.{3,}\s*\d{1,4}\s*$",
        "", text, flags=re.MULTILINE
    )

    legal_patterns = [
        r"^.*?this (document|report|message) is (confidential|proprietary).*?$",
        r"^.*?intended solely for.*?$",
        r"^.*?if you (have received|are not the intended).*?$",
        r"^.*?all rights reserved.*?$",
        r"^.*?unauthorized (use|reproduction|distribution).*?$",
        r"^.*?terms and conditions.*?$",
    ]
    for pattern in legal_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    text = re.sub(
        r"^\s*(figure|fig\.|table|chart|graph|diagram|appendix)\s+[\d\.]+[:\-]?.*?$",
        "", text, flags=re.MULTILINE | re.IGNORECASE
    )

    text = re.sub(
        r"^[A-Za-z][a-zA-Z\s\-]{2,40},\s*\d+(,\s*\d+)+\s*$",
        "", text, flags=re.MULTILINE
    )

    text = re.sub(
        r"^\s*(see also|refer to|see section|for more information see).*?$",
        "", text, flags=re.MULTILINE | re.IGNORECASE
    )

    return text


#  Normalizing abbreviations and numbers

def normalize_text(text: str) -> str:

    abbrevs = {
        r"\be\.g\.\s":    "for example ",
        r"\bi\.e\.\s":    "that is ",
        r"\betc\.\s":     "etcetera ",
        r"\bvs\.\s":      "versus ",
        r"\bDr\.\s":      "Dr ",
        r"\bMr\.\s":      "Mr ",
        r"\bMrs\.\s":     "Mrs ",
        r"\bMs\.\s":      "Ms ",
        r"\bProf\.\s":    "Prof ",
        r"\bSt\.\s":      "St ",
        r"\bAve\.\s":     "Ave ",
        r"\bapprox\.\s":  "approximately ",
        r"\bref\.\s":     "reference ",
        r"\bfig\.\s":     "figure ",
        r"\beq\.\s":      "equation ",
        r"\bvol\.\s":     "volume ",
        r"\bno\.\s":      "number ",
        r"\bp\.\s":       "page ",
        r"\bpp\.\s":      "pages ",
    }
    for pattern, replacement in abbrevs.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)

    text = re.sub(r"(\d)\s+%", r"\1%", text)
    text = re.sub(r"(\d)\s+percent\b", r"\1%", text, flags=re.IGNORECASE)

    return text


#  Quality validation

def validate(text: str) -> dict:

    warnings = []

    word_count  = len(text.split())
    char_count  = len(text)
    sentences   = re.split(r"(?<=[.!?])\s+", text.strip())
    sent_count  = len([s for s in sentences if len(s.split()) > 3])
    est_tokens  = int(word_count / 0.75)

    if word_count < 10:
        warnings.append(
            "CRITICAL: text has fewer than 10 words. Cannot summarize."
        )
    elif word_count < 50:
        warnings.append(
            "Text is very short (<50 words). Summary quality will be low."
        )

    avg_sent = word_count / max(sent_count, 1)
    if avg_sent < 5:
        warnings.append(
            f"Average sentence length is {avg_sent:.1f} words. "
            "Text appears fragmented (lists or table cells). "
            "Summarizer works best on continuous prose."
        )

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    repeated = [
        (l, c) for l, c in Counter(lines).items()
        if c >= 3 and len(l) > 5
    ]
    if repeated:
        examples = ", ".join(
            f'"{l[:30]}" x{c}' for l, c in repeated[:3]
        )
        warnings.append(
            f"Repeated lines found (possible leftover header/footer): {examples}"
        )

    return {
        "ok":               word_count >= 10,
        "word_count":       word_count,
        "char_count":       char_count,
        "sentence_count":   sent_count,
        "estimated_tokens": est_tokens,
        "needs_chunking":   est_tokens > 900,
        "warnings":         warnings,
    }


# main pipeline  clean_text()

def clean_text(text: str, verbose: bool = False) -> str:

    steps = [
        ("Fix encoding",           fix_encoding),
        ("Remove headers/footers", remove_headers_footers),
        ("Fix line breaks",        fix_line_breaks),
        ("Remove noise",           remove_noise),
        ("Normalize whitespace",   normalize_whitespace),
        ("Fix punctuation",        fix_punctuation),
        ("Remove boilerplate",     remove_boilerplate),
        ("Normalize text",         normalize_text),
        ("Final whitespace pass",  normalize_whitespace),
    ]

    if verbose:
        print(f"\n{'─' * 54}")
        print(f"  Starting : {len(text.split()):,} words")
        print(f"{'─' * 54}")

    for name, fn in steps:
        text = fn(text)
        if verbose:
            print(f"  [{name:<26}]  {len(text.split()):>6,} words")

    if verbose:
        print(f"{'─' * 54}\n")

    return text


#  utility :  Read .txt with auto encoding detection

def read_txt(filepath: str) -> str:

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if path.suffix.lower() not in (".txt", ".text", ".md"):
        print(f"Warning: expected .txt, got '{path.suffix}'. Proceeding.")

    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252", "ascii"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    return path.read_text(encoding="utf-8", errors="replace")


#  command line interface

def print_report(label: str, text: str):
    r = validate(text)
    print(f"\n  {label}")
    print(f"  {'─' * 48}")
    print(f"  Words           : {r['word_count']:,}")
    print(f"  Characters      : {r['char_count']:,}")
    print(f"  Sentences       : {r['sentence_count']:,}")
    print(f"  Estimated tokens: {r['estimated_tokens']:,}")
    chunking = "YES - chunk before summarizing" if r["needs_chunking"] else "No - fits in one call"
    print(f"  Needs chunking  : {chunking}")
    if r["warnings"]:
        for w in r["warnings"]:
            print(f"  WARNING: {w}")
    else:
        print(f"  Status          : Clean - ready to summarize")


def main():

    doc = pymupdf.open("git-cheat-sheet-education.pdf")
    out = open("output.txt", "wb" )

    for page in doc: # iteratring over pages of doc
        text = page.get_text().encode("utf8") #getting text of a page and encoding using utf8, storing in text
        out.write(text) # writing or inserting text into output txt file
        out.write(bytes((12,))) # same as out.write(b"\f") which is a page breaker in utf8 says that page ended
    out.close()
    
    # file mode
    input_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\nReading: {input_path}")
    raw = read_txt(input_path)
    print_report("Before cleaning", raw)

    cleaned = clean_text(raw, verbose=True)
    print_report("After cleaning", cleaned)

    if output_path:
        Path(output_path).write_text(cleaned, encoding="utf-8")
        print(f"\nSaved cleaned file to: {output_path}")
    else:
        preview_len = 2000
        print(f"\n-- Cleaned Text (first {preview_len} chars) ----------------------")
        print(cleaned[:preview_len])
        if len(cleaned) > preview_len:
            print(f"\n... [{len(cleaned) - preview_len:,} more characters]")


if __name__ == "__main__":
    main()