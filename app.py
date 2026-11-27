
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    abort
)

from PIL import Image, UnidentifiedImageError
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from io import BytesIO

import sqlite3
import base64
import binascii
import re
import json

from services.predictor import predict_acne, analyze_side_view


# ==================================================
# 1. KONFIGURASI APLIKASI
# ==================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

DB_PATH = BASE_DIR / "acnemap.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 18 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

MAX_IMAGE_SIZE = 5 * 1024 * 1024

ZONE_NAMES = [
    "dahi",
    "pipi_kiri",
    "pipi_kanan",
    "hidung",
    "dagu",
    "area_mulut",
    "tidak_terpetakan"
]


# ==================================================
# 2. DATABASE
# ==================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                prediction_json TEXT
            )
        """)

        columns = {
            row[1] for row in conn.execute(
                "PRAGMA table_info(analyses)"
            ).fetchall()
        }

        migrations = {
            "prediction_json": "TEXT",
            "left_filename": "TEXT",
            "right_filename": "TEXT",
            "profile_json": "TEXT"
        }

        for column, data_type in migrations.items():
            if column not in columns:
                conn.execute(
                    f"ALTER TABLE analyses ADD COLUMN {column} {data_type}"
                )


def save_analysis(
    filename,
    left_filename=None,
    right_filename=None
):
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO analyses (
                filename,
                left_filename,
                right_filename,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                filename,
                left_filename,
                right_filename,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )

        return cursor.lastrowid

init_db()


# ==================================================
# 3. FORMAT HASIL PREDIKSI DEFAULT
# ==================================================

def default_prediction(
    status="prediction_error",
    message="Analisis belum dapat dilakukan."
):

    return {
        "status": status,
        "model_name": None,
        "message": message,
        "face_detected": False,
        "detections": [],
        "total_lesions": None,

        "counts_by_zone": {
            zone: None for zone in ZONE_NAMES
        },

        "annotated_image": None,

        # CNN Severity
        "severity": None,
        "severity_confidence": None,
        "severity_description": None,
        "severity_error": None
    }


# ==================================================
# 4. VALIDASI DAN PENYIMPANAN GAMBAR
# ==================================================

def save_image_bytes(image_bytes):

    if not image_bytes:
        raise ValueError(
            "Foto kosong. Silakan ambil foto kembali."
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError(
            "Ukuran foto maksimal 5 MB."
        )

    try:

        # Memastikan file merupakan gambar valid.
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        # Membuka ulang gambar dan mengubah ke RGB.
        with Image.open(BytesIO(image_bytes)) as image:

            image = image.convert("RGB")

            filename = f"{uuid4().hex}.jpg"

            save_path = UPLOAD_DIR / filename

            image.save(
                save_path,
                format="JPEG",
                quality=90
            )

        return filename

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        raise ValueError(
            "Foto tidak valid. Gunakan JPG, JPEG, atau PNG."
        )


def save_uploaded_image(file):

    if not file or not file.filename:
        raise ValueError(
            "Pilih foto terlebih dahulu."
        )

    if "." not in file.filename:
        raise ValueError(
            "Format file tidak dikenali."
        )

    extension = file.filename.rsplit(".", 1)[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Gunakan gambar JPG, JPEG, atau PNG."
        )

    image_bytes = file.read()

    return save_image_bytes(image_bytes)


def save_camera_image(data_url):

    if not data_url:
        raise ValueError(
            "Ambil foto terlebih dahulu."
        )

    match = re.fullmatch(
        r"data:image/jpeg;base64,([A-Za-z0-9+/]*={0,2})",
        data_url
    )

    if not match:
        raise ValueError(
            "Format foto kamera tidak valid. "
            "Silakan ambil foto ulang."
        )

    encoded = match.group(1)

    if len(encoded) > 7 * 1024 * 1024:
        raise ValueError(
            "Ukuran foto kamera terlalu besar."
        )

    try:

        image_bytes = base64.b64decode(
            encoded,
            validate=True
        )

    except (ValueError, binascii.Error):

        raise ValueError(
            "Foto kamera gagal dibaca. "
            "Silakan ambil foto ulang."
        )

    return save_image_bytes(image_bytes)


# ==================================================
# 5. HALAMAN UTAMA
# ==================================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        saved_files = []

        photo_fields = [
            ("photo_front", "camera_photo_front"),
            ("photo_left", "camera_photo_left"),
            ("photo_right", "camera_photo_right")
        ]

        try:

            filenames = []

            for upload_field, camera_field in photo_fields:

                camera_data = request.form.get(
                    camera_field, ""
                ).strip()

                uploaded_file = request.files.get(
                    upload_field
                )

                if camera_data:
                    filename = save_camera_image(camera_data)

                elif uploaded_file and uploaded_file.filename:
                    filename = save_uploaded_image(uploaded_file)

                else:
                    raise ValueError(
                        "Lengkapi foto wajah depan, samping kiri, dan samping kanan."
                    )

                filenames.append(filename)
                saved_files.append(filename)

            analysis_id = save_analysis(
                filename=filenames[0],
                left_filename=filenames[1],
                right_filename=filenames[2]
            )

            return redirect(
                url_for(
                    "verification",
                    analysis_id=analysis_id
                )
            )

        except ValueError as error:

            # Membersihkan foto jika proses upload gagal.
            for filename in saved_files:
                (UPLOAD_DIR / filename).unlink(missing_ok=True)

            return render_template(
                "index.html",
                error=str(error)
            ), 400

    with get_db() as conn:
        total_analyses = conn.execute(
            "SELECT COUNT(*) FROM analyses"
        ).fetchone()[0]
        latest_analysis = conn.execute(
            "SELECT id, created_at FROM analyses ORDER BY id DESC LIMIT 1"
        ).fetchone()

    return render_template(
        "index.html",
        total_analyses=total_analyses,
        latest_analysis=latest_analysis
    )

# ==================================================
# 6. VERIFIKASI PROFIL KULIT
# ==================================================

@app.route(
    "/verification/<int:analysis_id>",
    methods=["GET", "POST"]
)
def verification(analysis_id):

    with get_db() as conn:
        analysis = conn.execute(
            """
            SELECT * FROM analyses
            WHERE id = ?
            """,
            (analysis_id,)
        ).fetchone()

    if analysis is None:
        abort(404, description="Data analisis tidak ditemukan.")

    if request.method == "POST":

        answers = request.get_json(silent=True)

        if not isinstance(answers, dict):
            return {
                "status": "error",
                "message": "Format jawaban tidak valid."
            }, 400

        required_fields = [
            "skin_type",
            "acne_frequency",
            "sensitivity",
            "main_concern"
        ]

        if any(
            not isinstance(answers.get(field), str)
            or not answers[field].strip()
            for field in required_fields
        ):
            return {
                "status": "error",
                "message": "Lengkapi semua pertanyaan terlebih dahulu."
            }, 400

        with get_db() as conn:
            conn.execute(
                """
                UPDATE analyses
                SET profile_json = ?
                WHERE id = ?
                """,
                (
                    json.dumps(answers, ensure_ascii=False),
                    analysis_id
                )
            )

        return {
            "status": "success",
            "redirect": url_for(
                "result",
                analysis_id=analysis_id
            )
        }

    return render_template(
        "verification.html",
        analysis=analysis
    )


# ==================================================
# 7. HALAMAN HASIL ANALISIS
# ==================================================



@app.route("/result/<int:analysis_id>")
def result(analysis_id):

    with get_db() as conn:
        analysis = conn.execute(
            """
            SELECT * FROM analyses
            WHERE id = ?
            """,
            (analysis_id,)
        ).fetchone()

    if analysis is None:
        abort(404, description="Riwayat analisis tidak ditemukan.")

    filename = analysis["filename"]
    full_image_path = UPLOAD_DIR / filename

    if not full_image_path.is_file():
        abort(404, description="File foto tidak ditemukan.")

    prediction = None

    if analysis["prediction_json"]:
        try:
            prediction = json.loads(analysis["prediction_json"])
        except (TypeError, json.JSONDecodeError):
            prediction = None

    # Jika belum pernah dianalisis, tampilkan status menunggu.
    if prediction is None:
        prediction = default_prediction(
            status="pending",
            message="Analisis belum dimulai. Tekan tombol Mulai Analisis."
        )

    image_url = url_for(
        "static",
        filename=f"uploads/{filename}"
    )

    return render_template(
        "result.html",
        image_path=image_url,
        analysis=analysis,
        prediction=prediction,
        message=prediction["message"]
    )


@app.route("/analyze/<int:analysis_id>", methods=["POST"])
def run_analysis(analysis_id):

    with get_db() as conn:
        analysis = conn.execute(
            """
            SELECT * FROM analyses
            WHERE id = ?
            """,
            (analysis_id,)
        ).fetchone()

    if analysis is None:
        abort(404, description="Riwayat analisis tidak ditemukan.")

    # Jangan menjalankan ulang analisis yang sudah berhasil.
    if analysis["prediction_json"]:
        try:
            existing_prediction = json.loads(
                analysis["prediction_json"]
            )

            if existing_prediction.get("status") == "success":
                return redirect(
                    url_for("result", analysis_id=analysis_id)
                )
        except (TypeError, json.JSONDecodeError):
            pass

    front_path = UPLOAD_DIR / analysis["filename"]

    if not front_path.is_file():
        abort(404, description="Foto depan tidak ditemukan.")

    try:
        # ANALISIS UTAMA: CNN1, YOLO, DAN CNN2.
        prediction = predict_acne(str(front_path))

        if not isinstance(prediction, dict):
            raise ValueError("Format hasil prediksi tidak valid.")

        defaults = default_prediction()

        for key, value in defaults.items():
            prediction.setdefault(key, value)

        if not isinstance(prediction.get("counts_by_zone"), dict):
            prediction["counts_by_zone"] = {}

        for zone in ZONE_NAMES:
            prediction["counts_by_zone"].setdefault(zone, None)

        if not isinstance(prediction.get("detections"), list):
            prediction["detections"] = []

        if prediction.get("total_lesions") is None:
            prediction["total_lesions"] = len(
                prediction["detections"]
            )

        # ANALISIS TAMBAHAN: FOTO KIRI DAN KANAN.
        prediction["side_view_analysis"] = {}

        side_view_files = {
            "left": analysis["left_filename"],
            "right": analysis["right_filename"]
        }

        for view_name, side_filename in side_view_files.items():

            if not side_filename:
                prediction["side_view_analysis"][view_name] = {
                    "status": "not_uploaded",
                    "message": "Foto samping tidak tersedia."
                }
                continue

            side_path = UPLOAD_DIR / side_filename

            if not side_path.is_file():
                prediction["side_view_analysis"][view_name] = {
                    "status": "error",
                    "message": "File foto samping tidak ditemukan."
                }
                continue

            try:
                prediction["side_view_analysis"][view_name] = (
                    analyze_side_view(str(side_path), view_name)
                )

            except Exception:
                app.logger.exception(
                    "Gagal menganalisis foto %s.",
                    view_name
                )

                prediction["side_view_analysis"][view_name] = {
                    "status": "error",
                    "message": "Analisis foto samping gagal."
                }

    except Exception:
        app.logger.exception(
            "Terjadi kesalahan saat menjalankan analisis."
        )

        prediction = default_prediction(
            status="prediction_error",
            message=(
                "Terjadi kesalahan saat menganalisis foto. "
                "Silakan coba kembali."
            )
        )

    # Simpan hasil analisis ke database.
    with get_db() as conn:
        conn.execute(
            """
            UPDATE analyses
            SET prediction_json = ?
            WHERE id = ?
            """,
            (
                json.dumps(prediction, ensure_ascii=False),
                analysis_id
            )
        )

    return redirect(
        url_for("result", analysis_id=analysis_id)
    )


# ==================================================
# 7. HALAMAN REKOMENDASI SKINCARE
# ==================================================



def verify_ingredient_safety(ingredient, profile):
    """
    Pemeriksaan awal kecocokan kandungan berdasarkan profil kulit.
    Bukan penilaian keamanan medis.
    """

    name = ingredient.get("name", "").lower()
    skin_type = str(profile.get("skin_type", "")).lower()
    sensitivity = str(profile.get("sensitivity", "")).lower()

    status = "Sesuai"
    reasons = []

    is_sensitive = (
        "sensitif" in sensitivity
        or sensitivity == "ya"
    )

    is_dry = "kering" in skin_type

    if is_sensitive and name in (
        "salicylic acid",
        "benzoyl peroxide",
        "azelaic acid"
    ):
        status = "Perlu Perhatian"
        reasons.append(
            "Kulit sensitif dapat lebih mudah mengalami "
            "kering, perih, atau iritasi dari bahan aktif ini."
        )

    if is_dry and name in (
        "salicylic acid",
        "benzoyl peroxide"
    ):
        status = "Perlu Perhatian"
        reasons.append(
            "Kulit kering perlu memperhatikan risiko "
            "kekeringan tambahan."
        )

    if name == "moisturizer" and (
        is_dry or is_sensitive
    ):
        reasons.append(
            "Pelembap menjadi bagian penting untuk "
            "menjaga kelembapan dan fungsi skin barrier."
        )

    if not reasons:
        reasons.append(
            "Tidak ditemukan peringatan khusus berdasarkan "
            "profil yang tersedia. Periksa komposisi lengkap "
            "produk dan perhatikan reaksi kulit."
        )

    ingredient["verification_status"] = status
    ingredient["verification_notes"] = reasons

    return ingredient


def build_skincare_guidance(prediction, profile=None):
    """
    Rekomendasi edukatif berdasarkan gabungan:
    1. Profil verifikasi kulit.
    2. Keparahan hasil CNN2.
    3. Zona wajah dengan jumlah lesi terbanyak.

    Sistem tidak menetapkan diagnosis atau resep medis.
    """

    prediction = prediction or {}
    profile = profile or {}

    # ==========================================
    # 1. MEMBACA HASIL VERIFIKASI
    # ==========================================

    skin_type = str(
        profile.get("skin_type", "")
    ).strip()

    acne_frequency = str(
        profile.get("acne_frequency", "")
    ).strip()

    sensitivity = str(
        profile.get("sensitivity", "")
    ).strip()

    main_concern = str(
        profile.get("main_concern", "")
    ).strip()

    skin_lower = skin_type.lower()
    frequency_lower = acne_frequency.lower()
    sensitivity_lower = sensitivity.lower()
    concern_lower = main_concern.lower()

    is_sensitive = (
        "sensitif" in sensitivity_lower
        or sensitivity_lower == "ya"
    )

    is_dry = "kering" in skin_lower
    is_oily = "berminyak" in skin_lower
    is_combination = "kombinasi" in skin_lower

    # ==========================================
    # 2. MEMBACA HASIL CNN2
    # ==========================================

    severity = prediction.get("severity")

    if isinstance(severity, str):
        severity = severity.lower().strip()
    else:
        severity = None

    try:
        total_number = int(
            prediction.get("total_lesions")
        )
    except (TypeError, ValueError):
        total_number = None

    # ==========================================
    # 3. MENGHITUNG ZONA DOMINAN
    # ==========================================

    counts = prediction.get("counts_by_zone") or {}

    zone_labels = {
        "dahi": "Dahi",
        "pipi_kiri": "Pipi kiri",
        "pipi_kanan": "Pipi kanan",
        "hidung": "Hidung",
        "dagu": "Dagu",
        "area_mulut": "Area sekitar mulut",
        "tidak_terpetakan": "Area belum terpetakan"
    }

    zone_factors = {
        "dahi": {
            "title": "Area dahi",
            "factors": [
                "Perhatikan kontak produk rambut dengan dahi.",
                "Bersihkan wajah setelah berkeringat.",
                "Perhatikan gesekan atau kondisi tertutup pada area dahi."
            ]
        },
        "pipi_kiri": {
            "title": "Area pipi kiri",
            "factors": [
                "Perhatikan kebersihan sarung bantal.",
                "Perhatikan gesekan masker atau benda lain.",
                "Hindari kebiasaan menyentuh wajah."
            ]
        },
        "pipi_kanan": {
            "title": "Area pipi kanan",
            "factors": [
                "Perhatikan kebersihan sarung bantal.",
                "Perhatikan gesekan masker atau benda lain.",
                "Hindari kebiasaan menyentuh wajah."
            ]
        },
        "hidung": {
            "title": "Area hidung",
            "factors": [
                "Area hidung memiliki banyak kelenjar minyak.",
                "Bersihkan wajah dengan lembut.",
                "Hindari menggosok atau memencet lesi."
            ]
        },
        "dagu": {
            "title": "Area dagu",
            "factors": [
                "Perhatikan kebiasaan menyentuh area dagu.",
                "Perhatikan gesekan masker.",
                "Catat pola kemunculan jerawat dari waktu ke waktu."
            ]
        },
        "area_mulut": {
            "title": "Area sekitar mulut",
            "factors": [
                "Perhatikan kecocokan produk bibir dan wajah.",
                "Hindari penggunaan produk yang memicu iritasi.",
                "Jangan menggosok area sekitar mulut."
            ]
        }
    }

    valid_counts = {}

    for zone, value in counts.items():
        try:
            count = int(value)
            if count >= 0:
                valid_counts[zone] = count
        except (TypeError, ValueError):
            continue

    zone_analysis = []

    for zone, count in valid_counts.items():
        if count <= 0:
            continue

        factor_data = zone_factors.get(zone)

        if not factor_data:
            continue

        zone_analysis.append({
            "zone": zone,
            "label": zone_labels.get(zone, zone),
            "count": count,
            "title": factor_data["title"],
            "factors": factor_data["factors"]
        })

    zone_analysis.sort(
        key=lambda item: item["count"],
        reverse=True
    )

    dominant_zone = (
        max(valid_counts, key=valid_counts.get)
        if valid_counts and max(valid_counts.values()) > 0
        else None
    )

    zone_label = zone_labels.get(
        dominant_zone,
        "Belum dapat ditentukan"
    )

    # ==========================================
    # 4. MENENTUKAN STATUS ANALISIS
    # ==========================================

    no_lesions = total_number == 0

    is_severe = (
    not no_lesions
    and severity == "severe"
)

    is_very_severe = (
    not no_lesions
    and severity == "very_severe"
)

    severe_condition = is_severe or is_very_severe

    if no_lesions:
        severity = None
        title = "Tidak ditemukan kandidat lesi jerawat"
        summary = (
            "Sistem tidak menemukan kandidat lesi pada foto. "
            "Tingkat keparahan jerawat tidak dapat ditentukan. "
            "Rekomendasi difokuskan pada kebutuhan dasar kulit "
            "dan masalah yang kamu pilih saat verifikasi."
        )

    elif severe_condition:
        title = "Utamakan evaluasi dokter kulit"
        summary = (
            "Hasil CNN2 menunjukkan kategori keparahan tinggi. "
            "Evaluasi dokter kulit disarankan untuk menentukan "
            "penanganan yang sesuai. Rekomendasi otomatis "
            "dibatasi pada perawatan dasar."
        )

    elif severity in ("mild", "moderate"):
        title = "Rekomendasi berdasarkan kondisi kulit"
        summary = (
            "Kandungan dipilih dengan mempertimbangkan profil "
            "kulit, frekuensi jerawat, masalah utama, tingkat "
            "keparahan CNN2, dan informasi zona dominan."
        )

    else:
        title = "Mulai dari perawatan dasar"
        summary = (
            "Klasifikasi keparahan belum tersedia. Sistem tidak "
            "akan menebak tingkat keparahan dan memprioritaskan "
            "perawatan dasar berdasarkan profil kulit."
        )

    # ==========================================
    # 5. KATALOG KANDUNGAN DAN SKOR PERSONAL
    # ==========================================

    ingredients = []

    def add_ingredient(
        name,
        tag,
        why,
        caution,
        evidence,
        reference,
        score
    ):
        ingredients.append({
            "name": name,
            "tag": tag,
            "why": why,
            "caution": caution,
            "evidence": evidence,
            "reference": reference,
            "_score": score
        })

    # Perawatan dasar untuk semua profil
    add_ingredient(
        "Gentle Cleanser",
        "Pembersih lembut",
        "Membantu membersihkan wajah tanpa menggosok kulit "
        "secara berlebihan.",
        "Hindari pembersih yang membuat kulit terasa tertarik "
        "atau perih.",
        "Perawatan pendukung",
        "https://www.nice.org.uk/guidance/ng198",
        100
    )

    add_ingredient(
        "Moisturizer",
        "Dukungan skin barrier",
        "Membantu menjaga kelembapan kulit dan mendukung "
        "fungsi pelindung kulit.",
        "Pilih formula yang nyaman dan dapat ditoleransi kulit.",
        "Perawatan pendukung",
        "https://www.nice.org.uk/guidance/ng198",
        95 if is_dry or is_sensitive else 70
    )

    add_ingredient(
        "Sunscreen SPF 30+",
        "Perlindungan harian",
        "Membantu melindungi kulit dari paparan UV dan "
        "mencegah noda pascajerawat semakin terlihat.",
        "Gunakan sesuai petunjuk produk dan ulangi pemakaian "
        "sesuai kebutuhan.",
        "Perawatan pendukung",
        "https://www.nice.org.uk/guidance/ng198",
        90 if "bekas jerawat" in concern_lower else 75
    )

    # ==========================================
    # 6. SKOR BERDASARKAN GABUNGAN INPUT
    # ==========================================

    # Kondisi tanpa lesi:
    # Tidak memberikan bahan aktif jerawat berdasarkan severity.
    if not no_lesions and not is_very_severe:

        acne_active = (
            "jerawat aktif" in concern_lower
        )

        acne_often = (
            "sering" in frequency_lower
            or "hampir selalu" in frequency_lower
        )

        acne_sometimes = (
            "sesekali" in frequency_lower
        )

        marks_concern = (
            "bekas jerawat" in concern_lower
        )

        redness_concern = (
            "kemerahan" in concern_lower
            or "iritasi" in concern_lower
        )

        texture_concern = (
            "tekstur" in concern_lower
        )

        # Salicylic Acid
        sa_score = 0

        if severity == "mild":
            sa_score += 1
        elif severity == "moderate":
            sa_score += 2

        if acne_active:
            sa_score += 2

        if acne_often:
            sa_score += 2
        elif acne_sometimes:
            sa_score += 1

        if is_oily:
            sa_score += 1

        if is_combination:
            sa_score += 1

        if is_dry:
            sa_score -= 2

        if is_sensitive:
            sa_score -= 2


        # Benzoyl Peroxide
        bpo_score = 0

        if severity == "moderate":
            bpo_score += 2

        if acne_active:
            bpo_score += 2

        if acne_often:
            bpo_score += 2

        if is_dry:
            bpo_score -= 2

        if is_sensitive:
            bpo_score -= 2


        # Azelaic Acid
        azelaic_score = 0

        if marks_concern:
            azelaic_score += 4

        if acne_active:
            azelaic_score += 1

        if redness_concern:
            azelaic_score += 1

        if severity in ("mild", "moderate"):
            azelaic_score += 1

        if is_sensitive:
            azelaic_score -= 1

        # Niacinamide
        niacinamide_score = 0

        if is_oily:
            niacinamide_score += 2

        if is_combination:
            niacinamide_score += 1

        if marks_concern:
            niacinamide_score += 1

        if redness_concern:
            niacinamide_score += 1

        if texture_concern:
            niacinamide_score += 1

        # Memasukkan kandungan yang memiliki skor relevan
        if sa_score > 0:
            add_ingredient(
                "Salicylic Acid",
                "Perawatan pori tersumbat",
                "Dipilih karena kombinasi profil kulit, frekuensi "
                "jerawat, tingkat keparahan, dan informasi area "
                "dominan menunjukkan relevansi untuk perawatan "
                "pori tersumbat. Lokasi bukan bukti penyebab jerawat.",
                "Dapat menyebabkan kulit kering atau iritasi. "
                "Mulai bertahap dan jangan langsung digabung "
                "dengan banyak bahan aktif.",
                "Rekomendasi bersyarat dalam pedoman AAD 2024",
                "https://pubmed.ncbi.nlm.nih.gov/38300170/",
                sa_score
            )

        if bpo_score > 0:
            add_ingredient(
                "Benzoyl Peroxide",
                "Perawatan jerawat",
                "Diprioritaskan ketika hasil keparahan, frekuensi "
                "jerawat, dan masalah jerawat aktif mendukung "
                "pertimbangan bahan ini.",
                "Dapat menyebabkan kering dan iritasi serta "
                "memudarkan kain. Ikuti petunjuk produk.",
                "Direkomendasikan dalam pedoman AAD 2024",
                "https://pubmed.ncbi.nlm.nih.gov/38300170/",
                bpo_score
            )

        if azelaic_score > 0:
            add_ingredient(
                "Azelaic Acid",
                "Jerawat dan noda pascajerawat",
                "Dipilih terutama ketika masalah utama adalah "
                "bekas jerawat, serta dapat menjadi pilihan "
                "perawatan jerawat sesuai toleransi kulit.",
                "Mulai bertahap. Hentikan penggunaan jika "
                "menimbulkan iritasi yang mengganggu.",
                "Didukung tinjauan sistematis",
                "https://pubmed.ncbi.nlm.nih.gov/37550898/",
                azelaic_score
            )

        if niacinamide_score > 0:
            add_ingredient(
                "Niacinamide",
                "Perawatan pendukung",
                "Dipilih sebagai bahan pendukung berdasarkan "
                "jenis kulit dan masalah utama. Dapat mendukung "
                "fungsi skin barrier.",
                "Bahan pendukung, bukan pengganti terapi utama "
                "untuk jerawat.",
                "Bukti klinis ditinjau dalam publikasi ilmiah",
                "https://pubmed.ncbi.nlm.nih.gov/38722460/",
                niacinamide_score
            )

    # ==========================================
    # 7. PENANGANAN BEKAS JERAWAT TANPA LESI
    # ==========================================

    if no_lesions and "bekas jerawat" in concern_lower:

        add_ingredient(
            "Azelaic Acid",
            "Perhatian pada noda pascajerawat",
            "Dipertimbangkan karena kamu memilih bekas jerawat "
            "sebagai masalah utama. Tidak adanya lesi pada foto "
            "tidak berarti noda pascajerawat sudah dipastikan.",
            "Gunakan bertahap dan perhatikan toleransi kulit. "
            "Jika kulit sensitif, prioritaskan perawatan dasar "
            "terlebih dahulu.",
            "Didukung tinjauan sistematis",
            "https://pubmed.ncbi.nlm.nih.gov/37550898/",
            3 if not is_sensitive else 1
        )

    # ==========================================
    # 8. BATASAN UNTUK KULIT SENSITIF
    # ==========================================

    active_items = [
        item for item in ingredients
        if item["name"] in (
            "Salicylic Acid",
            "Benzoyl Peroxide",
            "Azelaic Acid"
        )
    ]

    if is_sensitive and len(active_items) > 1:
        active_items.sort(
            key=lambda item: item["_score"],
            reverse=True
        )

        keep_name = active_items[0]["name"]

        ingredients = [
            item for item in ingredients
            if item["name"] not in (
                "Salicylic Acid",
                "Benzoyl Peroxide",
                "Azelaic Acid"
            )
            or item["name"] == keep_name
        ]

    # Kondisi berat: hanya perawatan dasar
    if is_very_severe:
        ingredients = [
        item for item in ingredients
        if item["name"] in (
            "Gentle Cleanser",
            "Moisturizer",
            "Sunscreen SPF 30+"
        )
    ]

    # ==========================================
    # 9. MENGURUTKAN KANDUNGAN
    # ==========================================

    ingredients.sort(
        key=lambda item: item["_score"],
        reverse=True
    )

    # Maksimal dua bahan aktif agar rekomendasi tidak berlebihan
    active_count = 0
    final_ingredients = []

    for item in ingredients:
        if item["name"] in (
            "Salicylic Acid",
            "Benzoyl Peroxide",
            "Azelaic Acid",
            "Niacinamide"
        ):
            active_count += 1

            if active_count > 2:
                continue

        checked = verify_ingredient_safety(
            item.copy(),
            profile
        )

        checked.pop("_score", None)
        final_ingredients.append(checked)

    ingredients = final_ingredients

    # ==========================================
    # 10. KETERANGAN PERSONALISASI
    # ==========================================

    personalization_note = (
        "Rekomendasi menggabungkan jenis kulit, frekuensi "
        "jerawat, sensitivitas, masalah utama, hasil keparahan "
        "CNN2, dan zona dominan."
    )

    if is_sensitive:
        personalization_note += (
            " Karena kulit terindikasi sensitif, jumlah bahan "
            "aktif dibatasi dan toleransi kulit menjadi prioritas."
        )

    if is_dry:
        personalization_note += (
            " Kondisi kulit kering membuat dukungan kelembapan "
            "menjadi prioritas."
        )

    if dominant_zone:
        personalization_note += (
            f" Area dengan jumlah kandidat lesi terbanyak adalah "
            f"{zone_label}. Informasi ini digunakan sebagai "
            f"konteks perawatan area, bukan untuk memastikan "
            f"penyebab jerawat."
        )

    if no_lesions:
        personalization_note += (
            " Karena tidak ditemukan kandidat lesi, sistem "
            "tidak memberikan rekomendasi bahan aktif jerawat "
            "berdasarkan tingkat keparahan."
        )

    if is_very_severe:
        personalization_note = (
        "Hasil CNN2 menunjukkan kategori Very Severe. "
        "Sistem memprioritaskan perawatan dasar dan "
        "menyarankan evaluasi dokter kulit sebelum "
        "mempertimbangkan penggunaan bahan aktif."
    )

    elif is_severe:
        personalization_note += (
        " Hasil CNN2 menunjukkan kategori Severe. "
        "Penggunaan bahan aktif perlu memperhatikan "
        "toleransi kulit dan disertai anjuran konsultasi "
        "dengan dokter kulit."
    )

    # ==========================================
    # 11. KETERANGAN DISTRIBUSI
    # ==========================================

    if no_lesions:
        distribution_note = (
            "Tidak ditemukan kandidat lesi pada foto. "
            "Tingkat keparahan jerawat tidak ditentukan."
        )

    elif total_number is not None and total_number > 0:
        distribution_note = (
            f"Terdapat {total_number} kandidat lesi yang "
            f"terdeteksi. Zona dengan jumlah tertinggi: "
            f"{zone_label}."
        )

    else:
        distribution_note = (
            "Jumlah kandidat lesi belum tersedia."
        )

    return {
        "severity": severity,
        "severity_label": {
            "mild": "Mild",
            "moderate": "Moderate",
            "severe": "Severe",
            "very_severe": "Very Severe"
        }.get(
            severity,
            "Tidak tersedia" if no_lesions
            else "Belum terklasifikasi"
        ),

        "title": title,
        "summary": summary,

        "ingredients": ingredients,
        "personalization_note": personalization_note,

        "skin_type": skin_type,
        "acne_frequency": acne_frequency,
        "sensitivity": sensitivity,
        "main_concern": main_concern,

        "distribution_note": distribution_note,
        "total_lesions": total_number,

        "dominant_zone": zone_label,
        "zone_analysis": zone_analysis,

        "has_analysis": bool(prediction),
        "has_zone_data": bool(zone_analysis),
        "no_lesions": no_lesions
    }



@app.route("/recommendations")
def recommendations():
    analysis_id = request.args.get("analysis_id", type=int)
    analysis = None
    prediction = None

    # Mengambil data analisis dari database
    with get_db() as conn:
        if analysis_id:
            analysis = conn.execute(
                "SELECT * FROM analyses WHERE id = ?",
                (analysis_id,)
            ).fetchone()
        else:
            analysis = conn.execute(
                "SELECT * FROM analyses ORDER BY id DESC LIMIT 1"
            ).fetchone()

    # Mengambil hasil prediksi CNN1 dan analisis lesi
    if analysis:
        if analysis["prediction_json"]:
            try:
                prediction = json.loads(analysis["prediction_json"])
            except (TypeError, json.JSONDecodeError):
                prediction = None

        # Jika hasil prediksi belum tersimpan, lakukan analisis
        if prediction is None:
            image_file = UPLOAD_DIR / analysis["filename"]

            if image_file.is_file():
                try:
                    prediction = predict_acne(str(image_file))

                    with get_db() as conn:
                        conn.execute(
                            """
                            UPDATE analyses
                            SET prediction_json = ?
                            WHERE id = ?
                            """,
                            (
                                json.dumps(
                                    prediction,
                                    ensure_ascii=False
                                ),
                                analysis["id"]
                            )
                        )

                except Exception:
                    app.logger.exception(
                        "Gagal memuat hasil untuk rekomendasi."
                    )
                    prediction = None

    # Mengambil jawaban kuesioner pengguna
    profile = {}

    if analysis and analysis["profile_json"]:
        try:
            profile = json.loads(analysis["profile_json"])
        except (TypeError, json.JSONDecodeError):
            profile = {}

    # Membuat rekomendasi berdasarkan hasil analisis dan profil pengguna
    guidance = build_skincare_guidance(
        prediction or {},
        profile
    )

    return render_template(
        "recommendations.html",
        guidance=guidance,
        analysis=analysis
    )

# ==================================================
# 8. HALAMAN RUTINITAS SKINCARE
# ==================================================

@app.route("/routine")
def routine():

    return render_template(
        "routine.html"
    )


# ==================================================
# 9. RIWAYAT ANALISIS
# ==================================================

@app.route("/history")
def history():

    with get_db() as conn:

        analyses = conn.execute(
            """
            SELECT * FROM analyses
            ORDER BY id DESC
            """
        ).fetchall()

    return render_template(
        "history.html",
        analyses=analyses
    )


# ==================================================
# 10. HAPUS RIWAYAT ANALISIS
# ==================================================

@app.route(
    "/history/delete/<int:analysis_id>",
    methods=["POST"]
)
def delete_analysis(analysis_id):

    with get_db() as conn:

        analysis = conn.execute(
            """
            SELECT * FROM analyses
            WHERE id = ?
            """,
            (analysis_id,)
        ).fetchone()

        if analysis is None:

            abort(
                404,
                description="Riwayat tidak ditemukan."
            )

        filename = analysis["filename"]

        # Menghapus catatan dari database.
        conn.execute(
            """
            DELETE FROM analyses
            WHERE id = ?
            """,
            (analysis_id,)
        )

        # Memeriksa apakah foto masih digunakan
        # oleh riwayat analisis lainnya.
        remaining = conn.execute(
            """
            SELECT COUNT(*)
            FROM analyses
            WHERE filename = ?
            """,
            (filename,)
        ).fetchone()[0]

        if remaining == 0:

            # Menghapus foto asli.
            image_path = (
                UPLOAD_DIR / filename
            ).resolve()

            if image_path.parent == UPLOAD_DIR.resolve():

                image_path.unlink(missing_ok=True)

            # Menghapus gambar anotasi terkait.
            annotated_name = (
                f"{Path(filename).stem}_annotated.jpg"
            )

            annotated_path = (
                RESULT_DIR / annotated_name
            ).resolve()

            if annotated_path.parent == RESULT_DIR.resolve():

                annotated_path.unlink(missing_ok=True)

    return redirect(
        url_for("history")
    )


# ==================================================
# 11. PENANGANAN ERROR
# ==================================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        "Data terlalu besar. Ukuran foto maksimal 5 MB.",
        413
    )


@app.errorhandler(404)
def page_not_found(error):

    if (BASE_DIR / "templates" / "404.html").exists():

        return render_template("404.html"), 404

    return (
        "Halaman atau data tidak ditemukan.",
        404
    )


# ==================================================
# 12. MENJALANKAN APLIKASI
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
