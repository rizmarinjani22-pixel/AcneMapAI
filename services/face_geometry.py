
import numpy as np


class FaceGeometry:
    """Menghitung sistem koordinat wajah untuk Zone Mapping V6."""

    def __init__(self, landmarks, width, height):
        def pt(i):
            p = landmarks[i]
            return np.array(
                [p.x * width, p.y * height],
                dtype=np.float32
            )

        self.pt = pt

        forehead_top = pt(10)
        chin_point = pt(152)
        left_eye = pt(33)
        right_eye = pt(263)
        nose_tip = pt(1)
        nose_left = pt(129)
        nose_right = pt(358)
        mouth_left = pt(61)
        mouth_right = pt(291)
        upper_lip = pt(13)
        lower_lip = pt(14)

        self.nose_tip_px = nose_tip
        self.face_left_px = pt(234)
        self.face_right_px = pt(454)

        self.eye_center = (left_eye + right_eye) / 2
        eye_vector = right_eye - left_eye
        self.eye_length = float(np.linalg.norm(eye_vector))

        if self.eye_length < 1:
            raise ValueError("Landmark mata tidak valid")

        self.h_axis = eye_vector / self.eye_length
        self.v_axis = np.array(
            [-self.h_axis[1], self.h_axis[0]],
            dtype=np.float32
        )

        if np.dot(self.v_axis, chin_point - self.eye_center) < 0:
            self.v_axis *= -1

        self.nose_u, self.nose_v = self.to_face(nose_tip)
        _, chin_v = self.to_face(chin_point)

        nose_left_u, _ = self.to_face(nose_left)
        nose_right_u, _ = self.to_face(nose_right)
        mouth_left_u, _ = self.to_face(mouth_left)
        mouth_right_u, _ = self.to_face(mouth_right)

        _, upper_lip_v = self.to_face(upper_lip)
        _, lower_lip_v = self.to_face(lower_lip)

        _, self.forehead_v = self.to_face(forehead_top)

        self.mouth_center_u = (
            mouth_left_u + mouth_right_u
        ) / 2

        self.eye_line_v = 0.0
        self.forehead_limit_v = self.eye_line_v - 0.25
        self.chin_limit_v = chin_v - 0.18

        self.nose_half_width = max(
            abs(nose_left_u - nose_right_u) / 2,
            0.10
        )

        self.mouth_half_width = max(
            abs(mouth_left_u - mouth_right_u) / 2,
            0.12
        )

        self.mouth_top = min(
            upper_lip_v, lower_lip_v
        ) - 0.10

        self.mouth_bottom = max(
            upper_lip_v, lower_lip_v
        ) + 0.10

    def to_face(self, p):
        relative = p - self.eye_center

        u = np.dot(relative, self.h_axis) / self.eye_length
        v = np.dot(relative, self.v_axis) / self.eye_length

        return float(u), float(v)

    def to_pixel(self, u, v):
        p = (
            self.eye_center
            + u * self.eye_length * self.h_axis
            + v * self.eye_length * self.v_axis
        )

        return int(round(p[0])), int(round(p[1]))

    def classify(self, cx, cy):
        u, v = self.to_face(
            np.array([cx, cy], dtype=np.float32)
        )

        if (
            self.mouth_top <= v <= self.mouth_bottom
            and abs(u - self.mouth_center_u)
            <= self.mouth_half_width * 1.5
        ):
            return "perioral", u, v

        if self.forehead_v <= v < self.forehead_limit_v:
            return "forehead", u, v

        if (
            abs(u - self.nose_u) <= self.nose_half_width
            and self.eye_line_v - 0.10
            <= v <= self.nose_v + 0.15
        ):
            return "nose", u, v

        if v > self.chin_limit_v:
            return "chin", u, v

        if self.eye_line_v - 0.05 <= v <= self.chin_limit_v:
            if u < self.nose_u - self.nose_half_width:
                return "person_left_cheek", u, v

            elif u > self.nose_u + self.nose_half_width:
                return "person_right_cheek", u, v

        return "unmapped", u, v

    def yaw_ratio(self):
        left_x = self.face_left_px[0]
        right_x = self.face_right_px[0]

        width = right_x - left_x

        if abs(width) < 1:
            return None

        return float(
            (
                self.nose_tip_px[0]
                - (left_x + right_x) / 2
            ) / width
        )
