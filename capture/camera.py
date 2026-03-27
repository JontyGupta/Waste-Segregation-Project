"""
Waste Classifier - Camera / Image Capture Module
Captures images from a webcam or loads from file using OpenCV.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class Camera:
    """
    Handles image acquisition from webcam or file system.

    Supports:
        - Live webcam capture (Single frame or continuous stream)
        - Loading images from disk
        - Basic validation of captured frames
    """

    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        save_dir: str = "data/captured",
        ) -> None:
        """
        Initialize Camera:

        Args:
            camera_index: Index of the webcam device (0 = default).
            frame_width: Camera width in pixels.
            frame_height: Capture height in pixels.
            save_dir: str = Directory to save captured images.
        """
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------------ #
    # Context manager support
    # ------------------------------------------------------------------------ #
    def __enter__(self) -> "Camera":
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ------------------------------------------------------------------------ #
    # Core methods
    # ------------------------------------------------------------------------ #
    def open(self) -> None:
        """Open the webcam device."""
        if self._cap is not None and self._cap.isOpened():
            logger.info("Camera already open.")
            return
        
        logger.info("Opening camera (index=%d)...", self.camera_index)
        self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.camera_index}."
                "Check that a webcam is connected."
            )
        
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_height)
        logger.info(
            "Camera opened (%dx%d).", self.frame_width, self.frame_height
        )

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame from the webcam.

        Returns:
            BGR image as numpy array.

        Raises:
            RuntimeError: If camera is not open or frame cannot be read.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not open. Call open() first.")
        
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from camera.")
        
        logger.debug("Frame captured - shape=%s", frame.shape)
        return frame
    
    def capture_and_save(self, filename:Optional[str] = None) -> Tuple[np.ndarray, str]:
        """
        Capture a frame and save it to disk.

        Args:
            filename: Optional filename. Auto-generated if not provided.

        Returns:
            Tuple of (frame, saved_file_path)
        """
        import time

        frame = self.capture_frame()
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"

        save_path = self.save_dir / filename
        cv2.imwrite(str(save_path), frame)
        logger.info("Frame saved to %s", save_path)
        return frame, str(save_path)
    
    def release(self) -> None:
        """"Release the webcam resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released.")

    # ------------------------------------------------------------------------ #
    # Static helpers
    # ------------------------------------------------------------------------ #
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load an image from disk

        Args:
            image_path: Path to the image file.

        Returns:
            BGR image as numpy array.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image cannot be decoded.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(path))
        if image is None:
            return ValueError(f"could not decode image: {image_path}")
        
        logger.debug("Loaded image %s - shape=%s", path.name, image.shape)
        return image
    
    @staticmethod
    def show_image(
        image: np.ndarray,
        window_name: str = "Waste Classifier",
        wait: bool = True,
    ) -> None:
        """
        Display an image in a window (useful for debugging).

        Args:
            image: BGR image to display.
            window_name: Title of the window.
            wait: If True, wait for a key presss to close.
        """
        cv2.imshow(window_name, image)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
