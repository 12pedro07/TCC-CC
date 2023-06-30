import cv2 # Image manipulation
import math # For some basic math operations
import json # For json file manipulation
import mosaic # Methods and dictionary for the mosaic construction
import logging # Log messages
import numpy as np # Numerical python
from tqdm import tqdm # Progress bar
from masks import Mask # Mask handling
from typing import Tuple, Callable # For type documentation
from pathlib import Path # Path management
from shapely import geometry # Geometry functions
from datetime import datetime, timedelta # Date and time manipulation
from functools import partial # Function manipulation
from insightface.app import FaceAnalysis # Face detection and points regression

class Pipeline:
    # ====== CONSTRUCTOR ====== #
    def __init__(self, **kwargs):
        # === Create attributes
        # String
        self.input_dataset: str = kwargs['input_dataset']
        self.output_identifier: str = self.input_dataset
        self.output_extension: str = kwargs['output_extension']
        # Integer
        self.ctx_id: int = kwargs['ctx_id']
        self.dot_size: int = kwargs['dot_size']
        self.bbox_size: int = kwargs['bbox_size']
        # Float
        self.mosaic_alpha: float = kwargs['mosaic_alpha']
        # Boolean
        self.show_numbers: bool = kwargs['show_numbers']
        self.partial_results: bool = kwargs['partial_results']
        # Tuple
        self.det_size: tuple[int] = kwargs['det_size']
        # Logger
        self.logger = logging.getLogger("Mosaic-Pipeline")
        # === Create path attributes
        self._create_paths()

    # ====== PIPELINE STEPS (PUBLIC) ====== #
    def detection(self) -> None:
        ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        # === Create the detection object
        self.app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'], providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'], root="/.insightface") # Detect and Use Regression for 106 2d keypoints
        self.app.prepare(ctx_id=self.ctx_id, det_size=(self.det_size, self.det_size)) # Set gpu/cpu and kernel size
        # === Read all images from the "sem_dor" and "com_dor" subfolders of the dataset
        for pain_class, imgs_path in [("sem_dor", self.path_nopain), ("com_dor", self.path_pain)]:
            for img_path in tqdm(imgs_path, desc=f'Detection: {pain_class}'):
                self.logger.debug(f'[STATUS] Processing detection for image {img_path.stem} ({pain_class})')
                # === Check and create path
                save_path = self.dst_path / img_path.stem
                save_path.mkdir(parents=True, exist_ok=True)
                # === Load image
                img = cv2.imread(str(img_path)) # Carrega a imagem com dor
                # === Detect face (get key points)
                img_marked, faces = self._detect_face(img)
                # === Check if no faces were detected
                if len(faces) == 0:
                    self.logger.warning(f'[WARNING] No faces found... skipping')
                    continue
                # === Assuming that only one face is present and ignore the others
                face = faces[0]
                bbox = face['bbox'].astype(int)
                bbox_shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                # === Crop the face
                self.logger.debug('[STATUS] Cropping face')
                face_img = self._crop_face(img, face['bbox'])
                # === Resize and create borders (if necessary)
                self.logger.debug('[STATUS] Resizing face image')
                cropped_img, borders, new_shape = self._resize_and_border(face_img)
                # === Rotate the face
                self.logger.debug('[STATUS] Rotating face image')
                # 1. Fix some params
                rotate = partial(self._resize_point, old_shape=bbox_shape, new_shape=new_shape, border_left=borders['left'], border_top=borders['top'])
                # 2. Change the origin from top left of the image to top left of the face
                mask2origin = self._change_reference(0,0, bbox[0], bbox[1])
                # 3. Translate the keypoints coordinates
                face_points = list(map(mask2origin, face['landmark_2d_106']))
                # 4. Resize and adjust the keypoints
                face_points = list(map(rotate, face_points)) # Redimensiona e reposiciona os pontos igual a bonding box
                # === Affine transform
                self.logger.debug('[STATUS] Applying affine transformation')
                # 1. Use the Left / Right eye coordinates and generate a third coordinate using a equilateral triangle
                src_3 = self._equilateral_triangle(face_points[35], face_points[93])
                # 2. Calculate the desired coordinates for the same points
                dst_1 = [self.bbox_size*0.2, self.bbox_size/3]
                dst_2 = [self.bbox_size*0.8, self.bbox_size/3]
                dst_3 = self._equilateral_triangle(dst_1, dst_2)
                # 3. Apply the transformation to the face image and save the transformation matrix
                cropped_img_affine, affine_matrix = self._affine_transform(
                    cropped_img, 
                    np.float32([face_points[35], face_points[93], src_3]), 
                    np.float32([dst_1, dst_2, dst_3])
                )
                # 4. Apply the transformation matrix on the keypoints
                face_points = cv2.transform(np.array([face_points]),affine_matrix)[0].tolist()
                # === Save results on a json file
                self.logger.debug('[STATUS] Saving data')
                with (save_path / "face_data.json").open('w') as ffp:
                    json.dump(
                        {
                            "timestamp": (datetime.utcnow()-timedelta(hours=3)).strftime("%Y-%m-%d_%H-%M-%S"), # Data_hora atual no Brasil
                            "label": pain_class,
                            "borders": borders,
                            "face_shape": new_shape,
                            "input_dataset": self.input_dataset,
                            "identifier": self.output_identifier,
                            **{key: value.astype(float).tolist() if isinstance(value, np.ndarray) else value.astype(float) for key, value in dict(face).items()},
                            "landmark_2d_106-crop_affine": face_points
                        }, 
                        ffp
                    )
                # === Save partial results
                if self.partial_results:
                    self.logger.debug('[STATUS] Saving partial results')
                    cv2.imwrite(str(save_path / f"detection{self.output_extension}"), img_marked) # Save Keypoints detection image
                    cv2.imwrite(str(save_path / f"bbox_crop{self.output_extension}"), cropped_img) # Save face image before affine transformation
                    cv2.imwrite(str(save_path / f"bbox_crop-affine{self.output_extension}"), cropped_img_affine) # Save face image after affine transformation 
    def mosaic(self) -> None:
        # === Iterate throught every detection
        files = list(self.dst_path.glob("*"))
        for file_path in tqdm(files, desc='Mosaic'):
            # === Implicit data
            face = file_path.stem
            face_results_path = self.dst_path / face
            # === Open the detection data
            try:
                with open(file_path / "face_data.json") as f:
                    face_data = json.load(f)
            except FileNotFoundError:
                if file_path.stem != 'Dataset':
                    self.logger.warning(f'.json file not found for {face}... skipping')
                continue
            # === Process mosaic
            # 1. Load the original image from the dataset
            face_path = Path("..", "Dataset", face_data['input_dataset']).glob(f"**/*{face}*")
            if self.partial_results:
                # Load original image
                try: 
                    face_path = next(face_path)
                except StopIteration: 
                    self.logger.warning(f"Face {face} not found on dataset... skipping")
                    continue
                img = cv2.imread(str(face_path))
                if img is None:
                    self.logger.warning(f"Erro loading image {face_path}... skipping")
                    continue
            # 2. Load the face image
            face_img = cv2.imread(str(face_results_path / f"bbox_crop-affine{self.output_extension}"))
            if face_img is None:
                self.logger.warning(f"Couldn't find affine transformation for {face}... skipping" )
                continue
            if self.partial_results: overlay = img.copy()
            overlay_face = face_img.copy()
            # 3. Load the mosaic regions and order by prority for drawing purposes only
            regions = list(mosaic.MOSAIC.items())
            regions.sort(key=lambda item1: item1[1]['priority'])
            # 4. Iterate on every region and draw it based on priority. The lower the priority, further it goes to the background
            for label, region_info in regions:
                # 4.1. Execute as simple mosaic (just connect the dots)
                try:
                    # Tenta executar como um mosaico
                    points = region_info["coords"]
                    # Separando e filtrando coordenadas da regiao do mosaico
                    if self.partial_results: points_filtered = [face_data['landmark_2d_106'][index] for index in points] # Pega a coordenada dos landmarks referente ao label atual
                    points_filtered_face = [face_data['landmark_2d_106-crop_affine'][index] for index in points] # Pega a coordenada dos landmarks referente ao label atual
                # If it doesn't have a group of dots to connect, consider it complex and call the associated function
                except KeyError:
                    try:
                        # Se nao conseguir, entao eh um mosaico complexo
                        if self.partial_results: points_filtered = region_info["function"](face_data['landmark_2d_106'])
                        points_filtered_face = region_info["function"](face_data['landmark_2d_106-crop_affine'])
                    except: # If it doesn't have a function either skip it
                        self.logger.warning(f'Region {label} has an invalid definition... skipping')
                        continue
                # 4.2. Dilate the polygon 
                points_filtered_face = self._dilate_polygon(points_filtered_face, pct=region_info['inflation'], dilate=True) # For the face only
                # 4.3. Reshape data and cat to int32
                points_filtered_face = np.array(points_filtered_face, dtype=np.int32).reshape((-1,1,2)) # Face image
                # 4.4. Create the necessary directories
                dataset_mask_path = self.output_dataset_path / "Mascaras" / face_data.get('label', 'desconhecido') / label
                dataset_crop_path = self.output_dataset_path / "Regioes" / face_data.get('label', 'desconhecido') / label
                dataset_mask_path.mkdir(parents=True, exist_ok=True)
                dataset_crop_path.mkdir(parents=True, exist_ok=True)
                # 4.5. Generate the binary mask
                mask = Mask.generate(
                    face_img.shape, # Height x Width
                    points_filtered_face, # Polygon coordinates
                    dataset_mask_path / f"{face}{self.output_extension}") # Output path
                # 4.6. Apply the mask to the image and save
                Mask.apply(
                    face_img,
                    mask,
                    dataset_crop_path / f"{face}{self.output_extension}")
                # 4.7. Save partial results
                if self.partial_results:
                    points_filtered = self._dilate_polygon(points_filtered, pct=region_info['inflation'], dilate=True) # For the complete image
                    points_filtered = np.array(points_filtered, dtype=np.int32).reshape((-1,1,2)) # Complete image
                    # Create the necessary directories
                    mask_file_path = face_results_path / "masks"
                    crop_file_path = face_results_path / "crops"
                    mask_file_path.mkdir(parents=True, exist_ok=True)
                    crop_file_path.mkdir(parents=True, exist_ok=True)
                    # Generate the binary mask
                    mask = Mask.generate(
                        face_img.shape, # Height x Width
                        points_filtered_face, # Polygon coordinates
                        mask_file_path / f"{label}{self.output_extension}") # Output path
                    # Apply the mask to the image and save
                    Mask.apply(
                        face_img,
                        mask,
                        crop_file_path / f"{label}{self.output_extension}")
                    # Draw the mask on the complete image
                    overlay = cv2.fillPoly(
                        overlay,                 # Image
                        [points_filtered],       # Polygon vertices
                        region_info["color"],    # Color
                        cv2.LINE_AA              # Line type
                    )
                    # Draw the mask on the face image
                    overlay_face = cv2.fillPoly(
                        overlay_face,
                        [points_filtered_face],
                        region_info["color"],
                        cv2.LINE_AA
                    )
            if self.partial_results: 
                # Apply the overlay
                face_img = cv2.addWeighted(overlay_face, self.mosaic_alpha, face_img, 1 - self.mosaic_alpha, 0)
                img = cv2.addWeighted(overlay, self.mosaic_alpha, img, 1 - self.mosaic_alpha, 0)
                # Save the result
                try:
                    cv2.imwrite(str(self.dst_path / ("mosaic_face-"+face)), face_img)
                    cv2.imwrite(str(self.dst_path / ("mosaic"+face)), img)
                except: pass

    # ====== PRIVATE METHODS ====== #
    def _affine_transform(self, img: np.array, src_points: Tuple[np.float32], dst_points: Tuple[np.float32]) -> Tuple[np.array, np.array]:
        rows, cols = img.shape[:2]
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
        img_output = cv2.warpAffine(img, affine_matrix, (cols,rows))
        return img_output, affine_matrix
    def _change_reference(self, old_x: int, old_y: int, new_x: int, new_y: int) -> Callable[[Tuple[int, int]], int]:
        delta_x = old_x + new_x
        delta_y = old_y + new_y
        return lambda coord: (coord[0]-delta_x, coord[1]-delta_y)
    def _crop_face(self, img: np.array, bbox: np.array) -> np.array:
        bboxes = bbox.astype(int) # Cast bbox keypoints to int (they are detected as float)
        face_img = img[bboxes[1]:bboxes[3], bboxes[0]:bboxes[2]].copy() # Crop the face
        return face_img
    def _detect_face(self, img: np.array) -> tuple:
        """
        Processa uma dada imagem, marcando o bounding box 
        da primeira face detectada e seus pontos fiduciais

        Input:
            - img: Imagem carregada como numpy array;
        Output:
            - Imagem com a bounding box e pontos fiduciais marcados; 
            - Faces: Lista com pontos resultantes de cada face detectada;
        """
        faces = self.app.get(img) # Detecta as faces
        rimg = img.copy()
        for face in faces:
            # LandMarK (lmk)
            lmk = face.landmark_2d_106
            lmk = np.round(lmk).astype(int)
            for i in range(lmk.shape[0]):
                p = tuple(lmk[i])
                if self.show_numbers:
                    cv2.putText(
                        rimg,                       # Image
                        str(i),                     # Text
                        p,                          # Left bottom text coordinate
                        cv2.FONT_HERSHEY_SIMPLEX,   # Font
                        0.2*self.dot_size,          # Font size
                        (255,0,0),                  # Color
                        1,                          # Line width
                        cv2.LINE_AA)                # Line type
                cv2.circle(
                    rimg,           # Imagem
                    p,              # Center coordinate
                    self.dot_size,  # Radius
                    (255,0,0),      # Color
                    -1,             # Line width (-1 = fill)
                    cv2.LINE_AA)    # Line type
        return (rimg, faces)
    def _equilateral_triangle(self, point1: Tuple[int], point2: Tuple[int]) -> Tuple[int, int]:
        x1, y1 = point1
        x2, y2 = point2

        dx = x2 - x1
        dy = y2 - y1

        alpha = 60./180*math.pi

        xp = x1 + math.cos( alpha)*dx + math.sin(alpha)*dy
        yp = y1 + math.sin(-alpha)*dx + math.cos(alpha)*dy

        return (xp, yp)
    def _resize_and_border(self, img: np.array) -> Tuple[np.array, Tuple[int], Tuple[int]] :
        old_size = img.shape[:2] # (height, width)
        ratio = float(self.bbox_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size]) # (width, height)
        img = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = self.bbox_size - new_size[1]
        delta_h = self.bbox_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        borders = {
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right
        }
        new_img = cv2.copyMakeBorder(
            img,                      
            top, bottom, left, right, 
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0])          
        return new_img, borders, (new_size[1], new_size[0])
    def _resize_point(self, point: Tuple[int], old_shape: Tuple[int], new_shape: Tuple[int], border_left: int, border_top: int) -> Tuple[int, int]:
        Ry = new_shape[1]/old_shape[1]
        Rx = new_shape[0]/old_shape[0]
        return (border_left + Rx * point[0], border_top + Ry * point[1])
    def _dilate_polygon(self, polygon, pct=0.10, dilate=True) -> np.array:
        polygon = geometry.Polygon(list(polygon))
        xs = list(polygon.exterior.coords.xy[0])
        ys = list(polygon.exterior.coords.xy[1])
        x_center = 0.5 * min(xs) + 0.5 * max(xs)
        y_center = 0.5 * min(ys) + 0.5 * max(ys)
        min_corner = geometry.Point(min(xs), min(ys))
        center = geometry.Point(x_center, y_center)
        shrink_distance = center.distance(min_corner)*pct
        if dilate: polygon_resized = polygon.buffer(shrink_distance) # dilate
        else: polygon_resized = polygon.buffer(-shrink_distance) # erode
        return np.array(list(zip(polygon_resized.exterior.coords.xy[0], polygon_resized.exterior.coords.xy[1])))

    # === SETUP PATHS
    def _create_paths(self) -> None:
        # Path attributes
        self.dst_path = Path("..", "Results", self.output_identifier)
        self.input_dataset_path = Path("..", "Dataset", self.input_dataset)
        self.path_nopain = list(self.input_dataset_path.joinpath("com_dor").glob('*')) # Generator for the files on "com_dor" directory
        self.path_pain = list(self.input_dataset_path.joinpath("sem_dor").glob('*')) # Generator for the files on "com_dor" directory
        self.output_dataset_path = self.dst_path.joinpath("Dataset")
        # Create directory
        self.output_dataset_path.mkdir(parents=True, exist_ok=True)