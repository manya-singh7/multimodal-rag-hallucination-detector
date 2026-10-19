
"""
Extract text and images from PDFs in data/raw/pdfs.

This script can be created and saved before downloading the PDFs.
It requires PyMuPDF when you are ready to run it.
"""

import json
import re
from pathlib import Path

PDF_DIR = Path("data/raw/pdfs")
TEXT_DIR = Path("data/processed/text")
IMAGE_DIR = Path("data/processed/images")


def split_into_chunks(text: str, max_words: int = 180) -> list[str]:
    """Split extracted text into manageable paragraph-based chunks."""
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []
    current_words = []

    for paragraph in paragraphs:
        words = paragraph.split()

        while words:
            room = max_words - len(current_words)

            if room <= 0:
                chunks.append(" ".join(current_words))
                current_words = []
                room = max_words

            current_words.extend(words[:room])
            words = words[room:]

            if len(current_words) >= max_words:
                chunks.append(" ".join(current_words))
                current_words = []

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in: {PDF_DIR.resolve()}")
        print("Add research-paper PDFs there before running this script.")
        return

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF is missing. Install it with:")
        print("python -m pip install pymupdf")
        return

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        document = fitz.open(pdf_path)
        all_chunks = []
        image_count = 0

        try:
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text")
                for chunk_number, chunk in enumerate(
                    split_into_chunks(page_text), start=1
                ):
                    all_chunks.append({
                        "id": (
                            f"{pdf_path.stem}_p{page_number}"
                            f"_c{chunk_number}"
                        ),
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": chunk,
                    })

                # Extract embedded images from each page.
                for image_number, image_info in enumerate(
                    page.get_images(full=True), start=1
                ):
                    xref = image_info[0]
                    extracted = document.extract_image(xref)
                    extension = extracted.get("ext", "png")
                    image_name = (
                        f"{pdf_path.stem}_p{page_number}"
                        f"_img{image_number}.{extension}"
                    )
                    image_path = IMAGE_DIR / image_name
                    image_path.write_bytes(extracted["image"])
                    image_count += 1

            output_path = TEXT_DIR / f"{pdf_path.stem}.json"
            output_path.write_text(
                json.dumps(all_chunks, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print(f"  Text chunks: {len(all_chunks)}")
            print(f"  Extracted images: {image_count}")
            print(f"  Saved text: {output_path}")

        finally:
            document.close()

    print("\nPDF processing finished.")


if __name__ == "__main__":
    main()
