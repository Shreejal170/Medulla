"""
FFmpeg-based frame extraction utilities.

This module provides low-level utilities for:
- Extracting frames from video files
- Normalizing and downscaling images
- Probing video metadata
- Cleaning up temporary frame directories
"""

import os
import subprocess
import logging
from pathlib import Path
from PIL import Image

from src.domain.models.analysis import ExtractedFrame, VideoExtractionData

logger = logging.getLogger("ffmpeg helper")

# Constants
MAX_DIMENSION = 512
TEMP_FRAMES_DIR = "temp_frames"


def extract_frames(
    video_id: str,
    file_path: str,
    sampling_fps: float,
    output_dir: str = TEMP_FRAMES_DIR,
    ffmpeg_bin: str = "ffmpeg"
) -> VideoExtractionData:
    """
    Extract frames from a video file using FFmpeg.
    
    Args:
        video_id: Unique identifier for the video
        file_path: Path to input video file
        sampling_fps: Target frames per second to extract
        output_dir: Directory to save extracted frames
        ffmpeg_bin: FFmpeg binary name or path (default: "ffmpeg")
        
    Returns:
        VideoExtractionData with extracted frame metadata
        
    Raises:
        FileNotFoundError: If video file not found
        RuntimeError: If FFmpeg execution fails or binary not found
        ValueError: If parameters are invalid
    """
    
    # Validation
    if not file_path:
        raise ValueError("Video file path cannot be empty")
    
    if sampling_fps <= 0:
        raise ValueError("sampling_fps must be positive")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise
    
    try:
        # Probe video to get original FPS
        original_fps = _probe_video_fps(file_path, ffmpeg_bin)
        if original_fps is None or original_fps <= 0:
            logger.warning("Could not determine video FPS, defaulting to 30")
            original_fps = 30
        
        # Extract frames
        extracted_frames = _extract_frames_subprocess(
            video_id=video_id,
            file_path=file_path,
            original_fps=original_fps,
            sampling_fps=sampling_fps,
            output_dir=output_dir,
            ffmpeg_bin=ffmpeg_bin
        )
        
        logger.info(f"Successfully extracted {len(extracted_frames)} frames from {video_id}")
        
        return VideoExtractionData(
            video_id=video_id,
            extracted_frames=extracted_frames,
            audio_path=None
        )
    
    except (FileNotFoundError, ValueError):
        raise
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else "Unknown error"
        logger.error(f"FFmpeg execution failed: {stderr}")
        raise RuntimeError(f"FFmpeg extraction failed: {stderr}") from e
    except Exception as e:
        logger.error(f"Unexpected error during extraction: {e}")
        raise RuntimeError(f"Extraction error: {str(e)}") from e


def _extract_frames_subprocess(
    video_id: str,
    file_path: str,
    original_fps: float,
    sampling_fps: float,
    output_dir: str,
    ffmpeg_bin: str
) -> list[ExtractedFrame]:
    """
    Extract frames using FFmpeg subprocess execution.
    
    Returns:
        List of ExtractedFrame objects
    """
    frame_interval = max(int(original_fps / sampling_fps), 1)
    fps_filter = original_fps / frame_interval
    
    output_pattern = os.path.join(output_dir, f"{video_id}_%04d.jpg")
    
    logger.debug(
        f"Extracting frames from {file_path} "
        f"at {fps_filter:.2f} fps (original: {original_fps:.2f}, "
        f"sampling: {sampling_fps:.2f})"
    )
    
    try:
        cmd = [
            ffmpeg_bin,
            "-i", file_path,
            "-vf", f"fps={fps_filter}",
            "-q:v", "2",
            output_pattern
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            stderr = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"FFmpeg failed: {stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, stderr=result.stderr)
        
        logger.debug("FFmpeg extraction completed successfully")
    
    except FileNotFoundError:
        raise RuntimeError(
            f"FFmpeg binary '{ffmpeg_bin}' not found. "
            "Please install FFmpeg: brew install ffmpeg (macOS) or "
            "sudo apt-get install ffmpeg (Linux) or "
            "choco install ffmpeg (Windows)"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg extraction timeout (10 minutes exceeded)")
    
    # Collect extracted frames
    extracted_frames = []
    frame_index = 0
    
    try:
        output_dir_path = Path(output_dir)
        frame_files = sorted(
            output_dir_path.glob(f"{video_id}_*.jpg"),
            key=lambda x: int(x.stem.split("_")[-1])
        )
        
        logger.debug(f"Found {len(frame_files)} frame files")
        
        for frame_file in frame_files:
            frame_file_str = str(frame_file)
            
            # Normalize image (resize + color conversion)
            if not normalize_image(frame_file_str):
                logger.warning(f"Failed to normalize frame {frame_file_str}, but continuing...")
            
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


def normalize_image(image_path: str) -> bool:
    """
    Normalize extracted frame image.
    
    Operations:
    - Resize to max 512px (maintaining aspect ratio)
    - Convert to RGB color space
    - Save as JPEG with consistent quality
    
    Args:
        image_path: Path to extracted frame image
        
    Returns:
        True if successful, False if failed
    """
    try:
        img = Image.open(image_path)
        logger.debug(f"Loaded image {image_path}: {img.size} {img.mode}")
        
        # Convert to RGB
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                img = rgb_img
            else:
                img = img.convert('RGB')
            logger.debug("Converted color space to RGB")
        
        # Resize maintaining aspect ratio
        width, height = img.size
        
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            scale = min(MAX_DIMENSION / width, MAX_DIMENSION / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"Resized from {width}x{height} to {new_width}x{new_height}")
        
        # Save as JPEG
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


def _probe_video_fps(file_path: str, ffmpeg_bin: str = "ffmpeg") -> float:
    """
    Probe video file to determine frames per second.
    
    Returns:
        FPS value or None if probe fails
    """
    try:
        cmd = [
            ffmpeg_bin,
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
            if "/" in fps_str:
                num, den = map(float, fps_str.split("/"))
                fps = num / den if den != 0 else None
                logger.debug(f"Probed FPS: {fps_str} = {fps}")
                return fps
            return float(fps_str)
        
        return None
    
    except subprocess.TimeoutExpired:
        logger.warning(f"FFmpeg probe timeout for {file_path}")
        return None
    except Exception as e:
        logger.warning(f"Failed to probe video FPS: {e}")
        return None


def cleanup_temp_frames(output_dir: str = TEMP_FRAMES_DIR) -> None:
    """
    Clean up temporary extracted frame files.
    
    Args:
        output_dir: Directory containing temporary frames
        
    Raises:
        RuntimeError: If cleanup fails (except for missing directory)
    """
    try:
        if not os.path.exists(output_dir):
            logger.debug(f"Temp directory does not exist: {output_dir}")
            return
        
        output_path = Path(output_dir)
        frame_files = list(output_path.glob("*.jpg"))
        
        if not frame_files:
            logger.debug(f"No frame files found in {output_dir}")
            return
        
        for frame_file in frame_files:
            try:
                os.remove(frame_file)
                logger.debug(f"Deleted: {frame_file}")
            except OSError as e:
                logger.warning(f"Failed to delete {frame_file}: {e}")
        
        logger.info(f"Cleaned up {len(frame_files)} temporary frame files")
    
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise RuntimeError(f"Cleanup failed: {str(e)}") from e
