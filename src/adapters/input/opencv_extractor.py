import os
import uuid
import cv2

from ports.output.frame_extractor_port import (
    FrameExtractorPort
)

from domain.models.analysis import (
    ExtractedFrame,
    VideoExtractionData
)


class OpenCVExtractor(
    FrameExtractorPort
):

    def __init__(
        self,
        output_dir: str = "temp_frames"
    ):
        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    def extract(
        self,
        video_id: str,
        file_path: str,
        sampling_fps: float
    ) -> VideoExtractionData:

        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            raise ValueError(
                f"Cannot open video: {file_path}"
            )

        original_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if original_fps <= 0:
            original_fps = 30

        frame_interval = max(
            int(original_fps / sampling_fps),
            1
        )

        extracted_frames = []

        frame_index = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % frame_interval == 0:

                timestamp_sec = (
                    frame_index / original_fps
                )

                frame_id = (
                    f"{video_id}_"
                    f"{uuid.uuid4().hex[:8]}"
                )

                resized = self._resize_frame(
                    frame
                )

                output_path = os.path.join(
                    self.output_dir,
                    f"{frame_id}.jpg"
                )

                cv2.imwrite(
                    output_path,
                    resized
                )

                extracted_frames.append(
                    ExtractedFrame(
                        frame_id=f"{video_id}_frame_{frame_index}",
                        frame_file_path=output_path,
                        timestamp_sec=timestamp_sec

                    )
                )

            frame_index += 1

        cap.release()

        return VideoExtractionData(
            video_id=video_id,
            extracted_frames=extracted_frames,
            audio_path=None
        )

    def _resize_frame(
        self,
        frame,
        max_width: int = 512
    ):

        height, width = frame.shape[:2]

        if width <= max_width:
            return frame

        scale = max_width / width

        new_size = (
            int(width * scale),
            int(height * scale)
        )

        return cv2.resize(
            frame,
            new_size
        )
    