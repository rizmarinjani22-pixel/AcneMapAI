import numpy as np
 
# STATUS: DIKUNCI (V6 final). Nilai dipilih dari DATA DEV (dataran sapuan).
# Jangan diubah setelah melihat hasil data uji.
DEFAULT_PARAMS = {
    "top_margin": 0.15,
    "fh_lower_center": -0.108,
    "fh_lower_side": -0.25,
    "nose_bottom": 0.10,
    "mouth_chin_t": 0.40,
    "side_w_perioral": 1.35,   # celah kosong dev: 1,326 (perioral) .. 1,390 (pipi)
    "side_w_chin": 1.10,       # tengah celah bersih dev: 1,009 .. 1,181
}
 
 
def _lip_chin_v(geo):
    """v bibir bawah (landmark 14) dan ujung dagu (landmark 152); di-cache."""
    if not hasattr(geo, "_v_lip_chin"):
        v_lip = geo.to_face(geo.pt(14))[1]
        v_chin = geo.to_face(geo.pt(152))[1]
        geo._v_lip_chin = (v_lip, v_chin)
    return geo._v_lip_chin
 
 
def face_metrics(geo, cx, cy):
    """Kembalikan u, v, t, w_rel, in_center untuk titik (cx, cy) piksel.
    t      : 0 = bibir bawah, 1 = ujung dagu.
    w_rel  : jarak ke tengah mulut dalam satuan setengah lebar mulut.
    """
    u, v = geo.to_face(np.array([cx, cy], dtype=np.float32))
    v_lip, v_chin = _lip_chin_v(geo)
    denom = v_chin - v_lip
    t = (v - v_lip) / denom if denom > 1e-6 else None
    w_rel = abs(u - geo.mouth_center_u) / geo.mouth_half_width
    in_center = abs(u - geo.nose_u) <= geo.nose_half_width
    return u, v, t, w_rel, in_center
 
 
def classify_v6(geo, cx, cy, params=None):
    """Kembalikan (zona, u, v). Nama zona sama dengan V5."""
    p = DEFAULT_PARAMS if params is None else {**DEFAULT_PARAMS, **params}
    u, v, t, w_rel, in_center = face_metrics(geo, cx, cy)
 
    top_limit = geo.forehead_v - p["top_margin"]
    nose_bottom = geo.nose_v + p["nose_bottom"]
    lower = p["fh_lower_center"] if in_center else p["fh_lower_side"]
 
    # 1. Di atas dahi (rambut / luar wajah)
    if v < top_limit:
        return "unmapped", u, v
 
    # 2. Dahi (termasuk alis dan glabella)
    if v < lower:
        return "forehead", u, v
 
    # 3. Hidung: kolom tengah, dari batas bawah dahi sampai dasar hidung
    if in_center and v <= nose_bottom:
        return "nose", u, v
 
    # 4. Perioral dan dagu: di bawah dasar hidung, dibatasi lebar mulut.
    #    Di samping hanya dimulai setinggi bibir atas (seperti V5).
    # Lebar zona dibedakan: perioral (setinggi bibir) lebih lebar daripada dagu
    if t is not None and t < p["mouth_chin_t"]:
        side_limit = p["side_w_perioral"]
    else:
        side_limit = p["side_w_chin"]
    lower_face = (
        v > nose_bottom
        and w_rel <= side_limit
        and (in_center or v >= geo.mouth_top)
    )
    if lower_face and t is not None:
        return ("perioral" if t < p["mouth_chin_t"] else "chin"), u, v
 
    # 5. Pipi (nama mengikuti sisi GAMBAR, seperti V5)
    if u < geo.nose_u - geo.nose_half_width:
        return "person_left_cheek", u, v
    if u > geo.nose_u + geo.nose_half_width:
        return "person_right_cheek", u, v
 
    return "unmapped", u, v