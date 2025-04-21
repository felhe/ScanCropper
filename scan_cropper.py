import io
import math
import os
from typing import Union, List, Dict, Any

import cv2
import fitz  # PyMuPDF
import numpy as np

from .arg_parse import ArgParser
from .settings import Settings

os.environ['QT_QPA_PLATFORM'] = 'xcb'


class ScanCropper:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.errors = 0
        self.images = 0
        self.scans = 0

        if self.settings.write_output:
            os.makedirs(settings.output_dir, exist_ok=True)

    def _load_pdf_as_images(self, pdf_bytes: bytes) -> List[np.ndarray]:
        dpi = 600
        zoom = dpi / 72.0
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mat = fitz.Matrix(zoom, zoom)

        images = []
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            img = img[:, :, :3] if pix.n == 4 else img
            images.append(img)

        return images

    def _load_pdf_from_path(self, pdf_path: str) -> List[np.ndarray]:
        with open(pdf_path, "rb") as file:
            return self._load_pdf_as_images(file.read())

    def _read_image_from_stream(self, stream: io.IOBase) -> np.ndarray:
        file_bytes = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    def _get_candidate_regions(self, img: np.ndarray, contours: List[np.ndarray]) -> List[Any]:
        img_area = img.shape[0] * img.shape[1]
        roi = []

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            area = cv2.contourArea(box)
            if (area / img_area) > 0.05:
                roi.append((box, rect, area))

        return sorted(roi, key=lambda r: r[2], reverse=True)

    def _rotate_image(self, img: np.ndarray, angle: float, center: tuple) -> np.ndarray:
        (h, w) = img.shape[:2]
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR)

    def _rotate_box(self, box: np.ndarray, angle: float, center: tuple) -> np.ndarray:
        rad = -angle * self.settings.deg_to_rad
        sine, cosine = math.sin(rad), math.cos(rad)
        rot_box = []

        for point in box:
            dx, dy = point[0] - center[0], point[1] - center[1]
            rot_x = dx * cosine - dy * sine + center[0]
            rot_y = dx * sine + dy * cosine + center[1]
            rot_box.append([rot_x, rot_y])

        return np.array(rot_box, dtype=np.float32)

    def _get_center(self, box: np.ndarray) -> tuple:
        x_vals, y_vals = box[:, 0], box[:, 1]
        return (x_vals.max() + x_vals.min()) / 2, (y_vals.max() + y_vals.min()) / 2

    def _clip_scans(self, img: np.ndarray, candidates: List[Any]) -> List[np.ndarray]:
        scans = []

        for box, rect, _ in candidates:
            angle = rect[2]
            if angle < -45:
                angle += 90

            box = np.intp(box)
            center = self._get_center(box)
            rotated_img = self._rotate_image(img, angle, center)
            rotated_box = self._rotate_box(box, angle, center)

            x_vals = np.clip(rotated_box[:, 0].astype(int), 0, img.shape[1])
            y_vals = np.clip(rotated_box[:, 1].astype(int), 0, img.shape[0])

            try:
                cropped = rotated_img[min(y_vals):max(y_vals), min(x_vals):max(x_vals)]
                scans.append(cropped)
            except Exception as e:
                print(f"Error cropping scan: {e}")
                self.errors += 1

        return scans

    def _find_scans(self, img: np.ndarray) -> List[np.ndarray]:
        blurred = cv2.medianBlur(img, self.settings.blur)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.settings.thresh, self.settings.max, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = self._get_candidate_regions(img, contours)
        return self._clip_scans(img, candidates)

    def process_input(self, source: Union[str, io.IOBase], name_hint: str = "input") -> List[np.ndarray]:
        self.images += 1

        if isinstance(source, str):
            if source.lower().endswith('.pdf'):
                imgs = self._load_pdf_from_path(source)
            else:
                img = cv2.imread(source)
                imgs = [img] if img is not None else []

        elif hasattr(source, 'read'):
            header = source.read(4)
            source.seek(0)

            if header.startswith(b'%PDF'):
                imgs = self._load_pdf_as_images(source.read())
            else:
                img = self._read_image_from_stream(source)
                imgs = [img] if img is not None else []
        else:
            raise ValueError("Unsupported input type. Must be file path or file-like object.")

        all_scans = []
        for i, img in enumerate(imgs):
            if img is None:
                print("Invalid image data.")
                continue

            scans = self._find_scans(img)
            for j, scan in enumerate(scans):
                if scan is None or not scan.size:
                    print(f"Skipping empty scan {j} in {name_hint}")
                    continue

                all_scans.append(scan)
                self.scans += 1

                if self.settings.write_output:
                    ext = 'jpg' if self.settings.output_format == 'jpg' else 'png'
                    filename = f"{name_hint}_{i}_{j}.{ext}"
                    output_path = os.path.join(self.settings.output_dir, filename)
                    params = [int(cv2.IMWRITE_JPEG_QUALITY), 100] if ext == 'jpg' else []
                    cv2.imwrite(output_path, scan, params)
                    print(f"Saved scan to {output_path}")

        return all_scans

    def process_inputs(self, sources: List[Union[str, io.IOBase]]) -> Dict[str, List[np.ndarray]]:
        results = {}
        for source in sources:
            name_hint = os.path.basename(source) if isinstance(source, str) else "stream"
            results[name_hint] = self.process_input(source, name_hint=name_hint)
        return results


def main():
    settings = ArgParser.parse()
    cropper = ScanCropper(settings)
    input_files = [os.path.join(settings.input_dir, f) for f in os.listdir(settings.input_dir)]
    cropper.process_inputs(input_files)


if __name__ == '__main__':
    main()
