import pyceres
import pycolmap


ImageReaderOptions = pycolmap.ImageReaderOptions()
# ImageReaderOptions.camera_params = '767.3861511125845,767.5058656118406, 679.054265997005,543.646891684636'
ImageReaderOptions.camera_params = '600,600,599.5,339.5'
pycolmap.extract_features(database_path, image_dir,camera_mode = pycolmap.CameraMode.SINGLE, camera_model="PINHOLE",reader_options=ImageReaderOptions)