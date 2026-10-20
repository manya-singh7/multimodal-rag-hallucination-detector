
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
    print(f"\n=== {label} ===")
    print(result["answer"])
    print("\nPossible supporting sentence:")
    print(result.get("supporting_sentence", "N/A"))
    print("\nText evidence:")
    for chunk in result.get("used_text", []):
        print(f"- ({chunk.get('source', 'Unknown')}) "
              f"{chunk.get('text', '')[:100]}...")
    print("\nImages:")
    for img in result.get("used_images", []):
        print(f"- {img.get('source_doc', 'Unknown')}: "
              f"{img.get('path', '')}")


def sample_text_chunks():
    return [
        {
            "text": (
                "The company reported revenue of $5.2 million in 2023, "
                "up 18% year over year."
            ),
            "source": "annual_report_2023.pdf",
            "page": 1,
        },
        {
            "text": (
                "Growth was driven primarily by the new enterprise product line."
            ),
            "source": "annual_report_2023.pdf",
            "page": 2,
        },
    ]


def sample_images():
    paths = [
        (os.path.join(STUB_IMAGE_DIR, "chart.png"),
         "Revenue chart 2023", "lightblue"),
        (os.path.join(STUB_IMAGE_DIR, "diagram.png"),
         "System diagram", "lightgreen"),
    ]

    result = []
    for path, label, color in paths:
        make_stub_image(path, label, color)
        result.append(path)
    return result


def test_text_and_images():
    retrieved = {
        "results": sample_text_chunks(),
        "image_paths": sample_images(),
    }
    query = (
        "What was the company's revenue in 2023 and what drove its growth?"
    )
    print_result(
        "TEXT + IMAGES",
        generate_answer(query, retrieved),
    )


def test_text_only():
    retrieved = {
        "results": sample_text_chunks(),
        "image_paths": [],
    }
    query = (
        "What was the company's revenue in 2023 and what drove its growth?"
    )
    print_result(
        "TEXT ONLY",
        generate_answer(query, retrieved),
    )


def test_images_only():
    retrieved = {
        "results": [],
        "image_paths": sample_images(),
    }
    query = (
        "Describe the supplied images and state what can be established "
        "about the company's revenue."
    )
    print_result(
        "IMAGES ONLY",
        generate_answer(query, retrieved),
    )


if __name__ == "__main__":
    test_text_and_images()
    test_text_only()
    test_images_only()
