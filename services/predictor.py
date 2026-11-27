
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
import json

from PIL import Image, ImageDraw, UnidentifiedImageError
from torchvision import transforms
from ultralytics import YOLO

from cnn_model.model import AcneSeverityCNN
from model_cnn2.model import AcneLesionCNN
from services.face_geometry import FaceGeometry
from models.zone_classifier_v6 import classify_v6


# ==================================================
# 1. PATH PROJECT
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Model CNN
CNN_MODEL_PATH = BASE_DIR / "cnn_model" / "best_severity_cnn.pth"

CNN2_MODEL_PATH = (
    BASE_DIR
    / "model_cnn2"
    / "acnemap_cnn2_imagenet_best.pth"
)
FUSION_CONFIG_PATH = (
    BASE_DIR / "model_cnn2" / "severity_fusion_config.json"
)
# Model YOLO
YOLO_MODEL_PATH = BASE_DIR / "models" / "best (1).pt"

# FaceLandmarker
FACE_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

# Folder hasil anotasi
RESULT_DIR = BASE_DIR / "static" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. KONFIGURASI CNN
# ==================================================

DEVICE = torch.device("cpu")

IMAGE_SIZE = 224

CLASS_NAMES = [
    "mild",
    "moderate",
    "severe",
    "very_severe"
]

SEVERITY_DESCRIPTIONS = {
    "mild": (
        "Hasil klasifikasi menunjukkan kategori "
        "Mild atau tingkat keparahan ringan."
    ),
    "moderate": (
        "Hasil klasifikasi menunjukkan kategori "
        "Moderate atau tingkat keparahan sedang."
    ),
    "severe": (
        "Hasil klasifikasi menunjukkan kategori "
        "Severe atau tingkat keparahan berat."
    ),
    "very_severe": (
        "Hasil klasifikasi menunjukkan kategori "
        "Very Severe atau tingkat keparahan sangat berat."
    )
}


def get_cnn_transform():

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ==================================================
# 3. LOAD CNN SATU KALI
# ==================================================

cnn_model = None


def load_cnn_model():

    global cnn_model

    if cnn_model is not None:
        return cnn_model

    if not CNN_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint CNN tidak ditemukan: {CNN_MODEL_PATH}"
        )

    checkpoint = torch.load(
        CNN_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    loaded_classes = checkpoint.get(
        "class_names",
        CLASS_NAMES
    )

    if list(loaded_classes) != CLASS_NAMES:
        raise ValueError(
            "Urutan kelas checkpoint CNN tidak sesuai "
            "dengan konfigurasi aplikasi."
        )

    model = AcneSeverityCNN(
        num_classes=len(CLASS_NAMES)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    cnn_model = model

    print("CNN Severity berhasil dimuat.")
    print("Checkpoint:", CNN_MODEL_PATH)

    return cnn_model

# ==================================================
# LOAD CNN 2 - LESION SEVERITY
# ==================================================

cnn2_model = None


def load_cnn2_model():

    global cnn2_model

    if cnn2_model is not None:
        return cnn2_model

    if not CNN2_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint CNN 2 tidak ditemukan: {CNN2_MODEL_PATH}"
        )

    checkpoint = torch.load(
        CNN2_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    loaded_classes = checkpoint.get(
        "class_names",
        CLASS_NAMES
    )

    if list(loaded_classes) != CLASS_NAMES:
        raise ValueError(
            "Urutan kelas checkpoint CNN 2 tidak sesuai."
        )

    model = AcneLesionCNN(
        num_classes=len(CLASS_NAMES)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    cnn2_model = model

    print("CNN 2 Lesion Severity berhasil dimuat.")
    print("Checkpoint:", CNN2_MODEL_PATH)

    return cnn2_model


# ==================================================
# 4. PREDIKSI SEVERITY SELURUH WAJAH
# ==================================================

def predict_severity(image_path):

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Foto untuk CNN tidak ditemukan: {image_path}"
        )

    try:
        image = Image.open(image_path).convert("RGB")

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"Foto tidak dapat dibaca oleh CNN: {error}"
        )

    transform = get_cnn_transform()

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    model = load_cnn_model()

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_class.item()

    severity = CLASS_NAMES[predicted_index]

    confidence_value = confidence.item() * 100

    return {
        "severity": severity,
        "severity_confidence": round(confidence_value, 2),
        "severity_description": SEVERITY_DESCRIPTIONS[severity],
        "severity_probabilities": probabilities[0].cpu().tolist()
    }

# ==================================================
# PREDIKSI CNN 2 - SETIAP CROP LESI
# ==================================================

def predict_lesion_severity(lesion_crop):

    transform = get_cnn_transform()

    image_tensor = transform(lesion_crop)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    model = load_cnn2_model()

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_class.item()

    severity = CLASS_NAMES[predicted_index]

    return {
        "lesion_severity": severity,
        "lesion_severity_confidence": round(
            confidence.item() * 100,
            2
        ),
        "lesion_probabilities": probabilities[0].cpu().tolist()

    }
def aggregate_cnn2_probabilities(detections):

    lesion_probabilities = [
        detection.get("lesion_probabilities")
        for detection in detections
        if detection.get("lesion_probabilities") is not None
    ]

    if not lesion_probabilities:
        return None

    probabilities = np.asarray(
        lesion_probabilities,
        dtype=np.float64
    )

    if probabilities.ndim != 2:
        return None

    if probabilities.shape[1] != len(CLASS_NAMES):
        return None

    aggregate = probabilities.mean(axis=0)

    total = aggregate.sum()

    if total <= 0:
        return None

    aggregate = aggregate / total

    return aggregate.tolist()

def decide_final_severity(cnn2_probabilities, detections):
    """
    Menentukan severity akhir berdasarkan agregasi CNN2.
    CNN2 menjadi sumber keputusan utama.
    """

    if cnn2_probabilities is None:
        return {
            "status": "insufficient_data",
            "severity": None,
            "confidence": None,
            "probabilities": None,
            "total_valid_lesions": 0,
            "class_distribution": {},
            "message": "Prediksi CNN2 tidak tersedia."
        }

    probabilities = np.asarray(
        cnn2_probabilities,
        dtype=np.float64
    )

    if probabilities.shape != (len(CLASS_NAMES),):
        return {
            "status": "invalid_data",
            "severity": None,
            "confidence": None,
            "probabilities": None,
            "total_valid_lesions": 0,
            "class_distribution": {},
            "message": "Format probabilitas CNN2 tidak valid."
        }

    if not np.all(np.isfinite(probabilities)):
        return {
            "status": "invalid_data",
            "severity": None,
            "confidence": None,
            "probabilities": None,
            "total_valid_lesions": 0,
            "class_distribution": {},
            "message": "Probabilitas CNN2 mengandung nilai tidak valid."
        }

    if np.any(probabilities < 0) or probabilities.sum() <= 0:
        return {
            "status": "invalid_data",
            "severity": None,
            "confidence": None,
            "probabilities": None,
            "total_valid_lesions": 0,
            "class_distribution": {},
            "message": "Distribusi probabilitas CNN2 tidak valid."
        }

    probabilities = probabilities / probabilities.sum()

    # Menghitung jumlah lesi yang berhasil diklasifikasikan
    valid_detections = [
        item for item in detections
        if item.get("lesion_probabilities") is not None
    ]

    total_valid_lesions = len(valid_detections)

    if total_valid_lesions == 0:
        return {
            "status": "insufficient_data",
            "severity": None,
            "confidence": None,
            "probabilities": None,
            "total_valid_lesions": 0,
            "class_distribution": {},
            "message": "Tidak ada lesi yang berhasil diklasifikasikan CNN2."
        }

    # Menghitung distribusi kelas prediksi setiap lesi
    class_distribution = {
        name: 0 for name in CLASS_NAMES
    }

    for item in valid_detections:
        label = item.get("lesion_severity")

        if label in class_distribution:
            class_distribution[label] += 1

    # Menentukan kelas dengan probabilitas agregasi tertinggi
    predicted_index = int(np.argmax(probabilities))

    final_severity = CLASS_NAMES[predicted_index]

    confidence = float(
        probabilities[predicted_index] * 100
    )

    return {
        "status": "success",
        "severity": final_severity,
        "confidence": round(confidence, 2),
        "probabilities": probabilities.tolist(),
        "total_valid_lesions": total_valid_lesions,
        "class_distribution": class_distribution,
        "message": (
            "Severity akhir ditentukan berdasarkan "
            "agregasi probabilitas CNN2."
        )
    }


def check_prediction_consistency(
    cnn1_severity,
    cnn2_probabilities,
    total_lesions
):
    """
    Membandingkan hasil CNN1 dan agregasi CNN2.
    Tidak mengubah severity utama CNN1.
    """

    if cnn1_severity is None:
        return {
            "status": "insufficient_data",
            "is_consistent": None,
            "message": "Prediksi CNN1 tidak tersedia."
        }

    if total_lesions == 0:
        return {
            "status": "insufficient_data",
            "is_consistent": None,
            "message": (
                "YOLO tidak menemukan kandidat lesi. "
                "Hasil CNN1 belum dapat dibandingkan "
                "dengan agregasi CNN2."
            )
        }

    if cnn2_probabilities is None:
        return {
            "status": "insufficient_data",
            "is_consistent": None,
            "message": (
                "Hasil agregasi CNN2 tidak tersedia."
            )
        }

    cnn2_index = int(
        np.argmax(cnn2_probabilities)
    )

    cnn2_severity = CLASS_NAMES[cnn2_index]

    is_consistent = (
        cnn1_severity == cnn2_severity
    )

    if is_consistent:
        status = "consistent"
        message = (
            "Prediksi CNN1 dan agregasi CNN2 "
            "memiliki kelas dominan yang sama."
        )
    else:
        status = "disagreement"
        message = (
            "Terdapat perbedaan kelas dominan "
            "antara prediksi CNN1 dan CNN2."
        )

    return {
        "status": status,
        "is_consistent": is_consistent,
        "cnn1_severity": cnn1_severity,
        "cnn2_dominant_severity": cnn2_severity,
        "message": message
    }

def load_fusion_config():

    if not FUSION_CONFIG_PATH.is_file():
        return None

    try:
        with open(
            FUSION_CONFIG_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            config = json.load(file)

        alpha = float(config["alpha"])

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "Alpha harus berada pada rentang 0 sampai 1."
            )

        if config.get("validated") is not True:
            return None

        if config.get("class_names") != CLASS_NAMES:
            raise ValueError(
                "Urutan kelas konfigurasi fusion tidak sesuai."
            )

        return config

    except Exception as error:

        print("FUSION CONFIG ERROR:", str(error))

        return None
    
def apply_severity_fusion(cnn1_probabilities, cnn2_probabilities):

    config = load_fusion_config()

    if config is None:
        return {
            "fusion_validated": False,
            "fusion_alpha": None,
            "fusion_probabilities": None,
            "fusion_severity": None,
            "fusion_confidence": None
        }

    if cnn1_probabilities is None or cnn2_probabilities is None:
        return {
            "fusion_validated": False,
            "fusion_alpha": None,
            "fusion_probabilities": None,
            "fusion_severity": None,
            "fusion_confidence": None
        }

    alpha = float(config["alpha"])

    p1 = np.asarray(cnn1_probabilities, dtype=np.float64)
    p2 = np.asarray(cnn2_probabilities, dtype=np.float64)

    if p1.shape != (len(CLASS_NAMES),):
        return {"fusion_validated": False}

    if p2.shape != (len(CLASS_NAMES),):
        return {"fusion_validated": False}

    fused = alpha * p1 + (1 - alpha) * p2

    fused = fused / fused.sum()

    predicted_index = int(np.argmax(fused))

    predicted_severity = CLASS_NAMES[predicted_index]

    return {
        "fusion_validated": True,
        "fusion_alpha": alpha,
        "fusion_probabilities": fused.tolist(),
        "fusion_severity": predicted_severity,
        "fusion_confidence": round(
            float(fused[predicted_index]) * 100,
            2
        )
    }

# ==================================================
# 5. KONFIGURASI YOLO
# ==================================================

CONF_THRESHOLD = 0.10
NMS_IOU = 0.70
YOLO_IMAGE_SIZE = 640


if not YOLO_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model YOLO tidak ditemukan: {YOLO_MODEL_PATH}"
    )

yolo_model = YOLO(str(YOLO_MODEL_PATH))


# ==================================================
# 6. PEMETAAN NAMA ZONA
# ==================================================

ZONE_TRANSLATION = {
    "forehead": "dahi",
    "nose": "hidung",
    "perioral": "area_mulut",
    "chin": "dagu",
    "person_left_cheek": "pipi_kiri",
    "person_right_cheek": "pipi_kanan",
    "unmapped": "tidak_terpetakan"
}


# ==================================================
# 7. FACE LANDMARKER
# ==================================================

def create_face_landmarker():

    if not FACE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"FaceLandmarker tidak ditemukan: {FACE_MODEL_PATH}"
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(FACE_MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1
    )

    return mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    )


# ==================================================
# 8. FORMAT HASIL ERROR
# ==================================================

def error_result(message, counts_by_zone=None):

    if counts_by_zone is None:
        counts_by_zone = {
            "dahi": 0,
            "pipi_kiri": 0,
            "pipi_kanan": 0,
            "hidung": 0,
            "dagu": 0,
            "area_mulut": 0,
            "tidak_terpetakan": 0
        }

    return {
        "status": "error",
        "model_name": None,
        "message": str(message),
        "face_detected": False,
        "detections": [],
        "total_lesions": 0,
        "counts_by_zone": counts_by_zone,
        "annotated_image": None,

        "severity": None,
        "severity_confidence": None,
        "severity_description": None,
        "severity_error": str(message)
    }


# ==================================================
# 9. FUNGSI UTAMA PREDIKSI
# ==================================================



def predict_acne(image_path):

    image_path = Path(image_path)

    counts_by_zone = {
        "dahi": 0,
        "pipi_kiri": 0,
        "pipi_kanan": 0,
        "hidung": 0,
        "dagu": 0,
        "area_mulut": 0,
        "tidak_terpetakan": 0
    }

    detections = []

    cnn2_aggregate_probabilities = None
    cnn2_dominant_lesion_class = None
    cnn2_dominant_lesion_score = None

    decision_result = None
    consistency_result = None

    face_detected = False
    geometry = None

    # ----------------------------------------------
    # A. VALIDASI FOTO
    # ----------------------------------------------

    if not image_path.is_file():
        return error_result(
            f"Foto tidak ditemukan: {image_path}",
            counts_by_zone
        )

    try:
        image = Image.open(image_path).convert("RGB")

    except (UnidentifiedImageError, OSError) as error:
        return error_result(
            f"Foto tidak dapat dibaca: {error}",
            counts_by_zone
        )

    width, height = image.size

    # ----------------------------------------------
    # B. PREDIKSI CNN1 - WHOLE FACE
    # CNN1 hanya sebagai pembanding
    # ----------------------------------------------

    severity_result = {
        "severity": None,
        "severity_confidence": None,
        "severity_description": None,
        "severity_probabilities": None,
        "severity_error": None
    }

    try:
        severity_result = predict_severity(image_path)
        severity_result["severity_error"] = None

        print(
            "CNN1 SEVERITY:",
            severity_result["severity"]
        )

        print(
            "CNN1 CONFIDENCE:",
            severity_result["severity_confidence"]
        )

    except Exception as error:
        print("CNN1 ERROR:", str(error))
        severity_result["severity_error"] = str(error)

    # ----------------------------------------------
    # C. DETEKSI WAJAH
    # ----------------------------------------------

    try:
        rgb_array = np.array(image)

        rgb_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_array
        )

        with create_face_landmarker() as landmarker:
            face_result = landmarker.detect(rgb_image)

        face_detected = bool(
            face_result.face_landmarks
        )

        if face_detected:
            landmarks = face_result.face_landmarks[0]

            geometry = FaceGeometry(
                landmarks,
                width,
                height
            )

    except Exception as error:
        print("FACE LANDMARKER ERROR:", str(error))

        face_detected = False
        geometry = None

    # ----------------------------------------------
    # D. DETEKSI LESI MENGGUNAKAN YOLO
    # ----------------------------------------------

    try:
        results = yolo_model.predict(
            source=str(image_path),
            imgsz=YOLO_IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=NMS_IOU,
            device="cpu",
            verbose=False
        )

        boxes = results[0].boxes

        # ------------------------------------------
        # E. ZONE MAPPING V6 + CNN2
        # ------------------------------------------

        for box in boxes:

            # 1. Koordinat bounding box YOLO

            x1, y1, x2, y2 = [
                float(value)
                for value in box.xyxy[0].tolist()
            ]

            confidence = float(box.conf[0])

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # 2. Crop lesi untuk CNN2

            crop_x1 = max(0, int(x1))
            crop_y1 = max(0, int(y1))
            crop_x2 = min(width, int(x2))
            crop_y2 = min(height, int(y2))

            lesion_result = {
                "lesion_severity": None,
                "lesion_severity_confidence": None,
                "lesion_probabilities": None
            }

            if crop_x2 > crop_x1 and crop_y2 > crop_y1:

                lesion_crop = image.crop((
                    crop_x1,
                    crop_y1,
                    crop_x2,
                    crop_y2
                ))

                try:
                    lesion_result = predict_lesion_severity(
                        lesion_crop
                    )

                except Exception as error:
                    print(
                        "CNN2 LESION ERROR:",
                        str(error)
                    )

            # 3. Zone Mapping V6

            if geometry is not None:

                zone, u, v = classify_v6(
                    geometry,
                    cx,
                    cy
                )

            else:
                zone = "unmapped"
                u = None
                v = None

            zone_name = ZONE_TRANSLATION.get(
                zone,
                "tidak_terpetakan"
            )

            counts_by_zone[zone_name] += 1

            # 4. Simpan hasil deteksi

            detections.append({
                "class": "acne_lesion",

                **lesion_result,

                "confidence": round(
                    confidence,
                    4
                ),

                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                },

                "center": {
                    "x": round(cx, 2),
                    "y": round(cy, 2)
                },

                "zone": zone,
                "zone_name": zone_name,

                "face_coordinates": {
                    "u": (
                        round(float(u), 4)
                        if u is not None else None
                    ),
                    "v": (
                        round(float(v), 4)
                        if v is not None else None
                    )
                }
            })

        # ------------------------------------------
        # F. AGREGASI PROBABILITAS CNN2
        # ------------------------------------------

        cnn2_aggregate_probabilities = (
            aggregate_cnn2_probabilities(detections)
        )

        if cnn2_aggregate_probabilities is not None:

            dominant_index = int(
                np.argmax(cnn2_aggregate_probabilities)
            )

            cnn2_dominant_lesion_class = (
                CLASS_NAMES[dominant_index]
            )

            cnn2_dominant_lesion_score = round(
                float(
                    cnn2_aggregate_probabilities[dominant_index]
                ) * 100,
                2
            )

        # ------------------------------------------
        # G. DECISION LOGIC CNN2
        # CNN2 menjadi sumber severity utama
        # ------------------------------------------

        decision_result = decide_final_severity(
            cnn2_probabilities=cnn2_aggregate_probabilities,
            detections=detections
        )

        print(
            "CNN2 FINAL SEVERITY:",
            decision_result.get("severity")
        )

        print(
            "CNN2 FINAL CONFIDENCE:",
            decision_result.get("confidence")
        )

        print(
            "CNN2 VALID LESIONS:",
            decision_result.get("total_valid_lesions")
        )

        # ------------------------------------------
        # H. CHECK KONSISTENSI CNN1 DAN CNN2
        # ------------------------------------------

        total_lesions = len(detections)

        consistency_result = check_prediction_consistency(
            cnn1_severity=severity_result.get("severity"),
            cnn2_probabilities=cnn2_aggregate_probabilities,
            total_lesions=total_lesions
        )

        # ------------------------------------------
        # I. MEMBUAT GAMBAR ANOTASI
        # ------------------------------------------

        annotated = image.copy()

        draw = ImageDraw.Draw(annotated)

        for detection in detections:

            bbox = detection["bbox"]

            x1 = bbox["x1"]
            y1 = bbox["y1"]
            x2 = bbox["x2"]
            y2 = bbox["y2"]

            label = (
                f'{detection["zone_name"]} '
                f'({detection["confidence"]:.2f})'
            )

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

            draw.text(
                (x1, max(0, y1 - 18)),
                label,
                fill="red"
            )

        # ------------------------------------------
        # J. MENYIMPAN GAMBAR HASIL
        # ------------------------------------------

        output_name = (
            f"{image_path.stem}_annotated.jpg"
        )

        output_path = RESULT_DIR / output_name

        annotated.save(
            output_path,
            quality=95
        )

        # ------------------------------------------
        # K. HASIL AKHIR ANALISIS
        # ------------------------------------------

        if face_detected:
            message = "Analisis berhasil dilakukan."
        else:
            message = (
                "Deteksi lesi berhasil, tetapi wajah "
                "tidak terdeteksi. Zona lesi belum "
                "dapat dipetakan."
            )

        if decision_result.get("status") != "success":
            message += (
                " Klasifikasi severity CNN2 belum berhasil."
            )

        return {
            "status": "success",

            "model_name": (
                "CNN2 Lesion + CNN1 Comparator + YOLO11n + "
                "FaceLandmarker + Zone Mapping V6"
            ),

            "message": message,

            "face_detected": face_detected,

            "detections": detections,

            "total_lesions": total_lesions,

            "counts_by_zone": counts_by_zone,

            "annotated_image": (
                f"/static/results/{output_name}"
            ),

            # CNN1 - WHOLE FACE COMPARATOR

            "cnn1_severity": severity_result.get("severity"),

            "cnn1_severity_confidence": (
                severity_result.get("severity_confidence")
            ),

            "cnn1_severity_error": (
                severity_result.get("severity_error")
            ),

            # CNN2 - AGGREGATE LESION

            "cnn2_aggregate_probabilities": (
                cnn2_aggregate_probabilities
            ),

            "cnn2_dominant_lesion_class": (
                cnn2_dominant_lesion_class
            ),

            "cnn2_dominant_lesion_score": (
                cnn2_dominant_lesion_score
            ),

            # CONSISTENCY CHECK

            "consistency_check": consistency_result,

            # FINAL DECISION

            "final_decision": decision_result,

            # CNN2 SEBAGAI SEVERITY UTAMA

            "severity": decision_result.get("severity"),

            "severity_confidence": (
                decision_result.get("confidence")
            ),

            "severity_description": (
                SEVERITY_DESCRIPTIONS.get(
                    decision_result.get("severity")
                )
                if decision_result.get("severity")
                else None
            ),

            "severity_probabilities": (
                decision_result.get("probabilities")
            ),

            "severity_error": (
                None
                if decision_result.get("status") == "success"
                else decision_result.get("message")
            )
        }

    except Exception as error:

        print(
            "YOLO / ZONE MAPPING ERROR:",
            str(error)
        )

        return {
            **error_result(
                str(error),
                counts_by_zone
            ),

            **severity_result,

            "status": "error",

            "model_name": None,

            "face_detected": face_detected,

            "detections": detections,

            "total_lesions": len(detections)
        }



# ==================================================
# 10. ANALISIS FOTO SAMPING (KIRI / KANAN)
# ==================================================

def analyze_side_view(image_path, view_name):

    image_path = Path(image_path)

    if view_name not in ("left", "right"):
        return {
            "status": "error",
            "message": "Jenis foto samping tidak valid."
        }

    if not image_path.is_file():
        return {
            "status": "error",
            "message": "File foto samping tidak ditemukan."
        }

    try:
        image = Image.open(image_path).convert("RGB")

    except (UnidentifiedImageError, OSError) as error:
        return {
            "status": "error",
            "message": f"Foto tidak dapat dibaca: {error}"
        }

    width, height = image.size
    detections = []

    try:
        results = yolo_model.predict(
            source=str(image_path),
            imgsz=YOLO_IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=NMS_IOU,
            device="cpu",
            verbose=False
        )

        boxes = results[0].boxes

        for box in boxes:

            x1, y1, x2, y2 = [
                float(value)
                for value in box.xyxy[0].tolist()
            ]

            confidence = float(box.conf[0])

            crop_x1 = max(0, int(x1))
            crop_y1 = max(0, int(y1))
            crop_x2 = min(width, int(x2))
            crop_y2 = min(height, int(y2))

            lesion_result = {
                "lesion_severity": None,
                "lesion_severity_confidence": None,
                "lesion_probabilities": None
            }

            if crop_x2 > crop_x1 and crop_y2 > crop_y1:

                lesion_crop = image.crop((
                    crop_x1,
                    crop_y1,
                    crop_x2,
                    crop_y2
                ))

                try:
                    lesion_result = predict_lesion_severity(
                        lesion_crop
                    )

                except Exception as error:
                    print(
                        "CNN2 SIDE VIEW ERROR:",
                        str(error)
                    )

            detections.append({
                "class": "acne_lesion",
                **lesion_result,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })

        aggregate = aggregate_cnn2_probabilities(
            detections
        )

        dominant_class = None
        dominant_score = None

        if aggregate is not None:

            dominant_index = int(np.argmax(aggregate))

            dominant_class = CLASS_NAMES[dominant_index]

            dominant_score = round(
                float(aggregate[dominant_index]) * 100,
                2
            )

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for detection in detections:

            bbox = detection["bbox"]

            label = (
                f'{detection["lesion_severity"] or "CNN2 N/A"} '
                f'({detection["confidence"]:.2f})'
            )

            draw.rectangle(
                [
                    bbox["x1"],
                    bbox["y1"],
                    bbox["x2"],
                    bbox["y2"]
                ],
                outline="red",
                width=3
            )

            draw.text(
                (
                    bbox["x1"],
                    max(0, bbox["y1"] - 18)
                ),
                label,
                fill="red"
            )

        output_name = (
            f"{image_path.stem}_{view_name}_annotated.jpg"
        )

        output_path = RESULT_DIR / output_name

        annotated.save(output_path, quality=95)

        return {
            "status": "success",
            "view": view_name,
            "total_lesions": len(detections),
            "detections": detections,
           "cnn2_aggregate_probabilities": (
            aggregate
            if aggregate is not None else None
        ),
            "cnn2_dominant_lesion_class": dominant_class,
            "cnn2_dominant_lesion_score": dominant_score,
            "annotated_image": f"/static/results/{output_name}",
            "message": (
                "Analisis foto samping berhasil."
                if detections
                else "Tidak ada kandidat lesi yang terdeteksi YOLO."
            )
        }

    except Exception as error:

        print("SIDE VIEW ANALYSIS ERROR:", str(error))

        return {
            "status": "error",
            "view": view_name,
            "total_lesions": 0,
            "detections": [],
            "annotated_image": None,
            "message": str(error)
        }
