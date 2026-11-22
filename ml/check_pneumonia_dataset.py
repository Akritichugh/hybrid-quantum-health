from pathlib import Path

dataset_path = Path("ml/datasets/chest_xray")

for split in ["train", "test", "val"]:
    print(f"\n{split.upper()}")

    for category in ["NORMAL", "PNEUMONIA"]:
        folder = dataset_path / split / category

        images = [
            file for file in folder.iterdir()
            if file.suffix.lower() in [".jpeg", ".jpg", ".png"]
        ]

        print(f"{category}: {len(images)} images")