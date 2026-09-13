import os

from PIL import Image, ImageDraw

from generator import generate_answer

STUB_IMAGE_DIR = "test_assets"


def make_stub_image(path, text, color):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (400, 300), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill="black")
    img.save(path)


def print_result(label, result):
    print(f"=== {label}: ANSWER ===")
    print(result["answer"])
    print("\n=== SUPPORTING SENTENCE ===")
    print(result.get("supporting_sentence", "N/A"))
    print("\n=== USED TEXT ===")
    for chunk in result["used_text"]:
        print(f"- ({chunk['source_doc']}) {chunk['text'][:80]}...")
    print("\n=== USED IMAGES ===")
    for img in result["used_images"]:
        print(f"- ({img['source_doc']}) {img['path']} - {img['caption']}")
    print()


def test_text_and_images():
    img1_path = os.path.join(STUB_IMAGE_DIR, "chart.png")
    img2_path = os.path.join(STUB_IMAGE_DIR, "diagram.png")
    make_stub_image(img1_path, "Revenue chart 2023", "lightblue")
    make_stub_image(img2_path, "System diagram", "lightgreen")

    retrieved = {
        "text_chunks": [
            {
                "text": "The company reported revenue of $5.2 million in 2023, "
                        "up 18% year over year.",
                "source_doc": "annual_report_2023.pdf",
            },
            {
                "text": "Growth was driven primarily by the new enterprise product line.",
                "source_doc": "annual_report_2023.pdf",
            },
        ],
        "images": [
            {
                "path": img1_path,
                "caption": "Bar chart showing revenue by quarter in 2023",
                "source_doc": "annual_report_2023.pdf",
            },
            {
                "path": img2_path,
                "caption": "Diagram of the enterprise product architecture",
                "source_doc": "product_overview.pdf",
            },
        ],
    }

    query = "What was the company's revenue in 2023 and what drove its growth?"
    result = generate_answer(query, retrieved)
    print_result("TEXT + IMAGES", result)


def test_text_only():
    retrieved = {
        "text_chunks": [
            {
                "text": "The company reported revenue of $5.2 million in 2023, "
                        "up 18% year over year.",
                "source_doc": "annual_report_2023.pdf",
            },
            {
                "text": "Growth was driven primarily by the new enterprise product line.",
                "source_doc": "annual_report_2023.pdf",
            },
        ],
        "images": [],
    }

    query = "What was the company's revenue in 2023 and what drove its growth?"
    result = generate_answer(query, retrieved)
    print_result("TEXT ONLY", result)


def test_images_only():
    img1_path = os.path.join(STUB_IMAGE_DIR, "chart.png")
    img2_path = os.path.join(STUB_IMAGE_DIR, "diagram.png")
    make_stub_image(img1_path, "Revenue chart 2023", "lightblue")
    make_stub_image(img2_path, "System diagram", "lightgreen")

    retrieved = {
        "text_chunks": [],
        "images": [
            {
                "path": img1_path,
                "caption": "Bar chart showing revenue by quarter in 2023",
                "source_doc": "annual_report_2023.pdf",
            },
            {
                "path": img2_path,
                "caption": "Diagram of the enterprise product architecture",
                "source_doc": "product_overview.pdf",
            },
        ],
    }

    query = "What was the company's revenue in 2023 and what drove its growth?"
    result = generate_answer(query, retrieved)
    print_result("IMAGES ONLY", result)


def main():
    test_text_and_images()
    test_text_only()
    test_images_only()


if __name__ == "__main__":
    main()