import math as m


class Settings:
    def __init__(self, threads=0, thresh=230, blur=9, scale=0.8, input_dir='./', output_dir='./',
                 output_file_name_prefix='', manual_name=False, manual_metadata=False, output_format='png',
                 write_output=True):
        self.threads = threads
        self.thresh = thresh
        self.blur = blur
        self.scale = scale
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_file_name_prefix = output_file_name_prefix
        self.manual_name = manual_name
        self.manual_metadata = manual_metadata
        self.output_format = output_format
        self.image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        self.deg_to_rad = m.pi / 180
        self.write_output = write_output
        self.max = 255  # Thresholded max value (white).
