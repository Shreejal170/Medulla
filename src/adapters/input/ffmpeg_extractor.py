import os
import logging
from pathlib import Path
from PIL import Image

try:
    import ffmpeg
except ImportError as e:
    raise ImportError(
        "ffmpeg-python is required but not installed. "
        "Please run: pip install ffmpeg-python\n"
        "Note: You also need FFmpeg binary installed on your system. "
        "See docs/ffmpeg_setup.md for installation instructions."
    ) from e

from ports.output.frame_extractor_port import FrameExtractorPort
from domain.models.analysis import ExtractedFrame, VideoExtractionData


logger = logging.getLogger(__name__)


class FFmpegExtractor(FrameExtractorPort):
    """
    Frame extractor using FFmpeg via ffmpeg-python abstraction.
    
    Benefits of ffmpeg-python over raw subprocess:
    - Cleaner, more maintainable API
    - Platform-independent command building
    - Better error handling and reporting
    - Easier to test and mock
    
    Replaces OpenCV with FFmpeg for improved:
    - codec support
    - performance at scale
    - platform stability
    """

    def __init__(self, output_dir: str = "temp_frames"):
        """
        Initialize FFmpeg extractor.
        
        Args:
            output_dir: Directory to save extracted frames
            
        Raises:
            RuntimeError: If FFmpeg binary is not found in PATH
        """
        self.output_dir = output_dir
        self.max_dimension = 512  # Max width or height for normalization
        
        # Verify FFmpeg is available
        self._verify_ffmpeg_available()
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise
    
    def _verify_ffmpeg_available(self) -> None:
        """
        Verify that FFmpeg binary is available.
        
        Raises:
            RuntimeError: If FFmpeg is not found in PATH
        """
        try:
            ffmpeg.probe(None)
        except ffmpeg.Error as e:
            error_msg = (
                "FFmpeg binary not found in system PATH.\n"
                "Please install FFmpeg:\n"
                "  - macOS: brew install ffmpeg\n"
                "  - Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "  - Windows: choco install ffmpeg (or download from ffmpeg.org)\n"
                "See docs/ffmpeg_setup.md for detailed setup instructions."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            # ffmpeg.probe(None) might fail differently, but if we can't detect
            # ffmpeg availability, we should fail loudly
            logger.debug(f"FFmpeg availability check: {e}")
            # For now, we'll proceed and let actual operations fail if FFmpeg is missing

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
                extracted_frames=extracted_frames,
                audio_path=None
            )
        
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Extraction validation error for {video_id}: {e}")
            raise
        except ffmpeg.Error as e:
            error_msg = (
                f"FFmpeg execution failed: {e.stderr.decode() if e.stderr else str(e)}"
                if hasattr(e, 'stderr') else f"FFmpeg error: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except RuntimeError:
            # Re-raise RuntimeError from extraction logic
            raise
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
            probe = ffmpeg.probe(file_path, select_streams='v:0')
            
            # Extract video stream
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                logger.warning(f"No video stream found in {file_path}")
                return None
            
            # Get frame rate from r_frame_rate (rational format like "30000/1001")
            if 'r_frame_rate' in video_stream:
                fps_str = video_stream['r_frame_rate']
                if "/" in fps_str:
                    try:
                        num, den = map(float, fps_str.split("/"))
                        fps = num / den if den != 0 else None
                        logger.debug(f"Probed FPS: {fps_str} = {fps}")
                        return fps
                    except (ValueError, ZeroDivisionError):
                        logger.warning(f"Could not parse FPS value: {fps_str}")
                else:
                    try:
                        return float(fps_str)
                    except ValueError:
                        logger.warning(f"Could not parse FPS value: {fps_str}")
            
            # Fallback to avg_frame_rate if r_frame_rate not available
            if 'avg_frame_rate' in video_stream:
                fps_str = video_stream['avg_frame_rate']
                if "/" in fps_str:
                    try:
                        num, den = map(float, fps_str.split("/"))
                        fps = num / den if den != 0 else None
                        logger.debug(f"Using avg_frame_rate: {fps_str} = {fps}")
                        return fps
                    except (ValueError, ZeroDivisionError):
                        pass
            
            logger.warning(f"Could not determine FPS from probe data")
            return None
        
        except ffmpeg.Error as e:
            logger.warning(f"FFmpeg probe failed: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
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
        Extract frames using FFmpeg with fps filter via ffmpeg-python.
        
        Returns:
            List of ExtractedFrame objects
            
        Raises:
            RuntimeError: If FFmpeg execution fails
        """
        
        frame_interval = max(int(original_fps / sampling_fps), 1)
        fps_filter = original_fps / frame_interval
        
        # Output pattern for frame files
        output_pattern = os.path.join(
            self.output_dir,
            f"{video_id}_%04d.jpg"
        )
        
        logger.debug(
            f"Extracting frames from {file_path} "
            f"at {fps_filter:.2f} fps (original: {original_fps:.2f}, "
            f"sampling: {sampling_fps:.2f})"
        )
        
        try:
            # Use ffmpeg-python to build and execute the extraction pipeline
            (
                ffmpeg
                .input(file_path)
                .filter('fps', f'{fps_filter:.2f}')
                .output(output_pattern, q=2)  # q=2 for high quality
                .overwrite_output()  # Overwrite existing files
                .run(quiet=False, capture_stdout=False, capture_stderr=True)
            )
            
            logger.debug(f"FFmpeg extraction completed successfully")
        
        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else "Unknown error"
            logger.error(f"FFmpeg extraction failed: {stderr}")
            raise RuntimeError(f"FFmpeg frame extraction failed: {stderr}") from e
        except Exception as e:
            logger.error(f"Unexpected error during FFmpeg execution: {e}")
            raise RuntimeError(f"Frame extraction error: {str(e)}") from e
        
        # Scan output directory for generated frames
        extracted_frames = []
        frame_index = 0
        
        try:
            output_dir_path = Path(self.output_dir)
            frame_files = sorted(
                output_dir_path.glob(f"{video_id}_*.jpg"),
                key=lambda x: int(x.stem.split("_")[-1])
            )
            
            logger.debug(f"Found {len(frame_files)} frame files")
            
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
                        frame_file_path=frame_file_str,
                        timestamp_sec=timestamp_sec
                    )
                )
                frame_index += 1
            
            logger.debug(f"Collected {len(extracted_frames)} frame metadata entries")
        
        except Exception as e:
            logger.error(f"Failed to collect extracted frames: {e}")
            raise RuntimeError(f"Failed to collect frame metadata: {str(e)}") from e
        
        return extracted_frames
