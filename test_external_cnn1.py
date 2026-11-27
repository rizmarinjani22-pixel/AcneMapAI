
import csv
from pathlib import Path

from services.predictor import predict_severity, CLASS_NAMES


# ==========================================
# KONFIGURASI PENGUJIAN EKSTERNAL CNN1
# ==========================================

TEST_DIR = Path("external_test")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "external_cnn1_results.csv"


def main():

    image_files = sorted(
        list(TEST_DIR.glob("*.jpg")) +
        list(TEST_DIR.glob("*.jpeg")) +
        list(TEST_DIR.glob("*.png"))
    )

    if not image_files:
        print("Tidak ada foto ditemukan di external_test.")
        return

    results = []

    print("\n======================================")
    print("       EXTERNAL TEST - CNN1")
    print("======================================")

    for image_path in image_files:

        print(f"\nMemproses: {image_path.name}")

        try:
            prediction = predict_severity(image_path)

            probabilities = prediction["severity_probabilities"]

            result = {
                "image": image_path.name,
                "predicted_severity": prediction["severity"],
                "confidence": prediction["severity_confidence"]
            }

            for index, class_name in enumerate(CLASS_NAMES):
                result[f"prob_{class_name}"] = round(
                    probabilities[index] * 100, 2
                )

            results.append(result)

            print("Prediksi:", prediction["severity"])
            print("Confidence:", prediction["severity_confidence"], "%")

            print("Probabilitas:")
            for index, class_name in enumerate(CLASS_NAMES):
                print(
                    f"  {class_name}: "
                    f"{probabilities[index] * 100:.2f}%"
                )

        except Exception as error:
            print("Gagal memproses:", error)

    if results:

        fieldnames = list(results[0].keys())

        with open(
            OUTPUT_FILE,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(results)

        print("\n======================================")
        print("PENGUJIAN SELESAI")
        print("Hasil disimpan:", OUTPUT_FILE)
        print("======================================")


if __name__ == "__main__":
    main()
