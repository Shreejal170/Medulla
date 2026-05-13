import os
import subprocess
import logging
from pathlib import Path
from PIL import Image

from ports.output.frame_extractor_port import FrameExtractorPort
from domain.models.analysis import ExtractedFrame, VideoExtractionData


logger = logging.getLogger(__name__)


class FFmpegExtractor(FrameExtractorPort):
    """
    Frame extractor using FFmpeg.
    
    Replaces OpenCV with FFmpeg for improved:
    - codec support
    - performance at scale
    - platform stability
    """

    def __init__(self, output_dir: str = "temp_frames", ffmpeg_bin: str = "ffmpeg"):
        """
        Initialize FFmpeg extractor.
        
        Args:
            output_dir: Directory to save extracted frames
            ffmpeg_bin: Path to ffmpeg binary (default: system PATH)
        """
        self.output_dir = output_dir
        self.ffmpeg_bin = ffmpeg_bin
        self.max_dimension = 512  # Max width or height for normalization
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise

    def extract(
        self,
        video_id: str,
        file_path: str,
        sampling_fps: float
    ) -> VideoExtractionData:
        """
        Extract frames from video using FFmpeg.
        
        Args:
            video_id: Unique video identifier
            file_path: Path to input video file
            sampling_fps: Target sampling rate (frames per second)
            
        Returns:
            VideoExtractionData with extracted frame metadata
            
        Raises:
            FileNotFoundError: If video file not found
            RuntimeError: If FFmpeg execution fails
            ValueError: If sampling_fps is invalid
        """
        
        # Validation
        if not file_path:
            logger.error("Video file path is empty")
            raise ValueError("Video file path cannot be empty")
        
        if sampling_fps <= 0:
            logger.error(f"Invalid sampling_fps: {sampling_fps}")
            raise ValueError("sampling_fps must be positive")
        
        if not os.path.exists(file_path):
            logger.error(f"Video file not found: {file_path}")
            raise FileNotFoundError(f"Video file not found: {file_path}")

        try:
            # Probe video to get FPS
            original_fps = self._probe_video_fps(file_path)
            if original_fps is None or original_fps <= 0:
                logger.warning(f"Could not determine video FPS, defaulting to 30")
                original_fps = 30
            
            # Extract frames
            extracted_frames = self._extract_frames_ffmpeg(
                video_id=video_id,
                file_path=file_path,
                original_fps=original_fps,
                sampling_fps=sampling_fps
            )
            
            logger.info(f"Successfully extracted {len(extracted_frames)} frames from {video_id}")
            
            return VideoExtractionData(
                video_id=video_id,
                frames=extracted_frames,
                audio_path=None,
                total_frames_extracted=len(extracted_frames)
            )
        
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Extraction validation error for {video_id}: {e}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg execution failed: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"FFmpeg extraction failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during extraction for {video_id}: {e}")
            raise RuntimeError(f"Unexpected extraction error: {str(e)}") from e

    def _normalize_image(self, image_path: str) -> bool:
        """
        Normalize extracted frame image:
        - Resize to max 512px (maintaining aspect ratio)
        - Convert to RGB color space
        - Save as JPEG with consistent quality
        
        Args:
            image_path: Path to extracted frame image
            
        Returns:
            True if successful, False if failed
        """
        try:
            # Load image
            img = Image.open(image_path)
            logger.debug(f"Loaded image {image_path}: {img.size} {img.mode}")
            
            # Convert to RGB (handle RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    # Convert RGBA to RGB (discard alpha channel)
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                    img = rgb_img
                else:
                    # Convert any other mode to RGB
                    img = img.convert('RGB')
                logger.debug(f"Converted color space to RGB")
            
            # Resize maintaining aspect ratio
            width, height = img.size
            
            if width > self.max_dimension or height > self.max_dimension:
                # Calculate scale factor
                scale = min(self.max_dimension / width, self.max_dimension / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug(f"Resized from {width}x{height} to {new_width}x{new_height}")
            
            # Save as JPEG with consistent quality
            img.save(image_path, 'JPEG', quality=85, optimize=False)
            logger.debug(f"Normalized and saved: {image_path}")
            return True
        
        except Image.UnidentifiedImageError as e:
            logger.error(f"Corrupted image file {image_path}: {e}")
            return False
        except OSError as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error normalizing {image_path}: {e}")
            return False

    def _probe_video_fps(self, file_path: str) -> float:
        """
        Probe video file to get FPS using FFmpeg.
        
        Returns:
            FPS value or None if probe fails
        """
        try:
            cmd = [
                self.ffmpeg_bin,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1:noinheader=1",
                file_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
                check=False
            )
            
            if result.returncode == 0 and result.stdout:
                fps_str = result.stdout.decode().strip()
                # Handle frame rate like "30000/1001" (29.97 fps)
                if "/" in fps_str:
                    num, den = map(float, fps_str.split("/"))
                    return num / den if den != 0 else None
                return float(fps_str)
            
            return None
        
        except subprocess.TimeoutExpired:
            logger.warning(f"FFmpeg probe timeout for {file_path}")
            return None
        except Exception as e:
            logger.warning(f"Failed to probe video FPS: {e}")
            return None

    def _extract_frames_ffmpeg(
        self,
        video_id: str,
        file_path: str,
        original_fps: float,
        sampling_fps: float
    ) -> list[ExtractedFrame]:
        """
        Extract frames using FFmpeg with fps filter.
        
        Returns:
            List of ExtractedFrame objects
        """
        
        frame_interval = max(int(original_fps / sampling_fps), 1)
        fps_filter = original_fps / frame_interval
        
        # Output pattern for frame files
        output_pattern = os.path.join(
            self.output_dir,
            f"{video_id}_%04d.jpg"
        )
        
        cmd = [
            self.ffmpeg_bin,
            "-i", file_path,
            "-vf", f"fps={fps_filter}",
            "-q:v", "2",  # Quality: 2 (high quality)
            output_pattern
        ]
        
        logger.debug(f"Running FFmpeg: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            stderr = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"FFmpeg failed: {stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, stderr=result.stderr)
        
        # Scan output directory for generated frames
        extracted_frames = []
        frame_index = 0
        
        try:
            output_dir_path = Path(self.output_dir)
            frame_files = sorted(
                output_dir_path.glob(f"{video_id}_*.jpg"),
                key=lambda x: int(x.stem.split("_")[-1])
            )
            
            for frame_file in frame_files:
                frame_file_str = str(frame_file)
                
                # PREPROCESSING: Normalize image (resize + color conversion)
                if not self._normalize_image(frame_file_str):
                    logger.warning(f"Failed to normalize frame {frame_file_str}, but continuing...")
                    # Continue anyway - normalization failure doesn't stop the pipeline
                
                # Calculate timestamp based on frame index
                timestamp_sec = (frame_index * frame_interval) / original_fps
                
                extracted_frames.append(
                    ExtractedFrame(
                        frame_id=f"frame_{frame_index:04d}",
                        file_path=frame_file_str,
                        timestamp_sec=timestamp_sec
                    )
                )
                frame_index += 1
            
            logger.debug(f"Collected {len(extracted_frames)} frame metadata entries")
        
        except Exception as e:
            logger.error(f"Failed to collect extracted frames: {e}")
            raise RuntimeError(f"Failed to collect frame metadata: {str(e)}") from e
        
        return extracted_frames
