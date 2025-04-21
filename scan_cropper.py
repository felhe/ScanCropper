import io
import math
import os

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

    def convert_pdf_bytes_to_images(self, pdf_stream):
        dpi = 600
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        images = []

        for i in range(len(doc)):
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = doc.get_page_pixmap(i, matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                img = img[:, :, :3]
            images.append(img)

        return images

    def convert_pdf_path_to_images(self, pdf_path):
        with open(pdf_path, "rb") as f:
            return self.convert_pdf_bytes_to_images(f.read())

    def get_candidate_regions(self, img, contours):
        roi = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            roi.append([box, rect, cv2.contourArea(box)])
        roi = sorted(roi, key=lambda b: b[2], reverse=True)

        img_area = img.shape[0] * img.shape[1]
        return [b for b in roi if (b[2] / img_area) > 0.05]

    def rotate_image(self, img, angle, center):
        (h, w) = img.shape[:2]
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR)

    def rotate_box(self, box, angle, center):
        rad = -angle * self.settings.deg_to_rad
        sine = math.sin(rad)
        cosine = math.cos(rad)
        rotBox = []
        for p in box:
            p[0] -= center[0]
            p[1] -= center[1]
            rot_x = p[0] * cosine - p[1] * sine
            rot_y = p[0] * sine + p[1] * cosine
            p[0] = rot_x + center[0]
            p[1] = rot_y + center[1]
            rotBox.append(p)
        return np.array(rotBox)

    def get_center(self, box):
        x_vals = [i[0] for i in box]
        y_vals = [i[1] for i in box]
        return ((max(x_vals) + min(x_vals)) / 2, (max(y_vals) + min(y_vals)) / 2)

    def clip_scans(self, img, candidates):
        scans = []
        for roi in candidates:
            rect = roi[1]
            box = np.intp(roi[0])
            angle = rect[2]
            if angle < -45:
                angle += 90
            center = self.get_center(box)
            rotIm = self.rotate_image(img, angle, center)
            rotBox = self.rotate_box(box, angle, center)
            x_vals = [int(i[0]) for i in rotBox]
            y_vals = [int(i[1]) for i in rotBox]
            try:
                scans.append(rotIm[min(y_vals):max(y_vals), min(x_vals):max(x_vals)])
            except IndexError:
                print("Error: Cropping failed, likely due to bounds.")
                self.errors += 1
        return scans

    def find_scans(self, img):
        blur = cv2.medianBlur(img, self.settings.blur)
        grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(grey, self.settings.thresh, self.settings.max, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi = self.get_candidate_regions(img, contours)
        return self.clip_scans(img, roi)

    def read_image_from_stream(self, stream):
        file_bytes = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    def process_input(self, source, name_hint="input"):
        self.images += 1

        if isinstance(source, str):
            # File path
            if source.lower().endswith('.pdf'):
                imgs = self.convert_pdf_path_to_images(source)
            else:
                img = cv2.imread(source)
                imgs = [img] if img is not None else []
        elif isinstance(source, io.BytesIO) or hasattr(source, 'read'):
            # File-like object
            header = source.read(4)
            source.seek(0)
            if header.startswith(b'%PDF'):
                imgs = self.convert_pdf_bytes_to_images(source.read())
            else:
                img = self.read_image_from_stream(source)
                imgs = [img] if img is not None else []
        else:
            raise ValueError("Unsupported input type. Must be path or file-like object.")

        all_scans = []
        for i, img in enumerate(imgs):
            if img is None:
                print("Invalid image data.")
                continue

            scans = self.find_scans(img)
            for j, scan in enumerate(scans):
                if scan is None or not scan.size:
                    print(f"Skipping empty scan {j} in {name_hint}")
                    continue

                all_scans.append(scan)
                self.scans += 1

                if self.settings.write_output:
                    fname = f"{name_hint}_{i}_{j}"
                    ext = 'jpg' if self.settings.output_format == 'jpg' else 'png'
                    out_path = os.path.join(self.settings.output_dir, f"{fname}.{ext}")
                    params = [int(cv2.IMWRITE_JPEG_QUALITY), 100] if ext == 'jpg' else []
                    cv2.imwrite(out_path, scan, params)
                    print(f"Saved scan to {out_path}")

        return all_scans

    def process_inputs(self, sources: list):
        results = {}
        for source in sources:
            name_hint = os.path.basename(source) if isinstance(source, str) else "stream"
            scans = self.process_input(source, name_hint=name_hint)
            results[name_hint] = scans
        return results


def main():
    settings = ArgParser.parse()
    cropper = ScanCropper(settings)
    input_paths = [os.path.join(settings.input_dir, f) for f in os.listdir(settings.input_dir)]
    cropper.process_inputs(input_paths)


if __name__ == '__main__':
    main()
