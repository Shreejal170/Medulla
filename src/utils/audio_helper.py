"""
Audio extraction and spectrogram generation utilities.

This module provides low-level utilities for:
- Extracting audio from video files at standardized format (mono, 16kHz, WAV)
- Chunking audio into temporal segments with metadata
- Generating Mel spectrograms from audio chunks for LLM-based audio analysis
- Cleaning up temporary audio files and spectrograms
"""

import os
import subprocess
import logging
import shutil
from pathlib import Path
from typing import List, Tuple
import numpy as np

try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for headless environments
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.domain.models.analysis import ExtractedAudioChunk, AudioExtractionData

logger = logging.getLogger("audio_helper")

# Constants for audio processing
TEMP_AUDIO_DIR = "temp_audio"
TEMP_AUDIO_CHUNKS_DIR = "temp_audio_chunks"
TEMP_SPECTROGRAMS_DIR = "temp_spectrograms"

# Audio standardization parameters
STANDARD_SAMPLE_RATE = 16000  # 16 kHz for speech/audio analysis
STANDARD_CHANNELS = 1  # Mono
STANDARD_FORMAT = "wav"

# Chunking parameters
DEFAULT_CHUNK_DURATION_SEC = 5  # 5-second chunks

# Spectrogram parameters
MEL_BINS = 128  # Number of mel frequency bins
FFT_SIZE = 2048  # FFT window size
HOP_LENGTH = 512  # Number of samples between successive frames
SPECTROGRAM_DPI = 100  # Resolution for saved spectrograms


def extract_audio(
    video_id: str,
    file_path: str,
    output_dir: str = TEMP_AUDIO_DIR,
    ffmpeg_bin: str = "ffmpeg"
) -> str:
    """
    Extract audio from video file and standardize to mono, 16kHz WAV format.

    Standardization ensures consistent audio representation for downstream
    processing and LLM analysis. Using a fixed sample rate and channels
    simplifies spectrogram generation and model input consistency.

    Args:
        video_id: Unique identifier for the video
        file_path: Path to input video file
        output_dir: Directory to save extracted audio
        ffmpeg_bin: FFmpeg binary name or path (default: "ffmpeg")

    Returns:
        Path to extracted audio file

    Raises:
        FileNotFoundError: If video file not found
        RuntimeError: If FFmpeg execution fails or binary not found
        ValueError: If parameters are invalid
    """

    # Validation
    if not file_path:
        raise ValueError("Video file path cannot be empty")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise

    try:
        # Generate deterministic output filename based on video_id
        audio_file = os.path.join(
            output_dir,
            f"{video_id}.{STANDARD_FORMAT}"
        )

        # If audio already extracted, return existing file
        if os.path.exists(audio_file):
            logger.info(f"Audio file already exists: {audio_file}")
            return audio_file

        logger.debug(
            f"Extracting audio from {file_path} "
            f"(mono, {STANDARD_SAMPLE_RATE}Hz, {STANDARD_FORMAT})"
        )

        # FFmpeg command to extract and standardize audio
        # -ac 1: Convert to mono (1 audio channel)
        # -ar 16000: Resample to 16 kHz
        # -acodec pcm_s16le: PCM 16-bit format for WAV
        cmd = [
            ffmpeg_bin,
            "-i", file_path,
            "-ac", str(STANDARD_CHANNELS),
            "-ar", str(STANDARD_SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            "-y",  # Overwrite output file if exists
            audio_file
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            stderr = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"FFmpeg audio extraction failed: {stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, stderr=result.stderr)

        logger.info(f"Successfully extracted audio to: {audio_file}")
        return audio_file

    except FileNotFoundError:
        raise RuntimeError(
            f"FFmpeg binary '{ffmpeg_bin}' not found. "
            "Please install FFmpeg: brew install ffmpeg (macOS) or "
            "sudo apt-get install ffmpeg (Linux) or "
            "choco install ffmpeg (Windows)"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg audio extraction timeout (10 minutes exceeded)")
    except Exception as e:
        logger.error(f"Unexpected error during audio extraction: {e}")
        raise RuntimeError(f"Audio extraction error: {str(e)}") from e


def chunk_audio(
    audio_file_path: str,
    video_id: str,
    chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
    output_dir: str = TEMP_AUDIO_CHUNKS_DIR
) -> List[ExtractedAudioChunk]:
    """
    Split audio file into temporal chunks with metadata.

    Chunking aligns with frame-level architecture: just as frame extraction
    produces multiple temporal segments for visual analysis, audio chunking
    creates temporal segments for audio analysis. This enables:
    - Parallel processing of audio chunks (future Kafka pipeline)
    - Consistent temporal granularity with video frames
    - Memory efficiency (avoid processing entire audio at once)
    - Per-segment LLM inference (similar to frame-level analysis)

    Args:
        audio_file_path: Path to extracted audio file
        video_id: Unique identifier for the video
        chunk_duration_sec: Duration of each chunk in seconds (default: 5)
        output_dir: Directory to save audio chunks

    Returns:
        List of ExtractedAudioChunk objects with paths and metadata

    Raises:
        FileNotFoundError: If audio file not found
        RuntimeError: If audio loading fails
        ValueError: If chunk duration is invalid
    """

    if not HAS_LIBROSA:
        raise RuntimeError(
            "librosa not installed. Install with: pip install librosa"
        )

    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    if chunk_duration_sec <= 0:
        raise ValueError("chunk_duration_sec must be positive")

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise

    try:
        logger.debug(f"Loading audio file: {audio_file_path}")

        # Load audio using librosa
        # sr=None preserves original sample rate
        y, sr = librosa.load(audio_file_path, sr=STANDARD_SAMPLE_RATE)
        total_duration = librosa.get_duration(y=y, sr=sr)

        logger.debug(
            f"Loaded audio: duration={total_duration:.2f}s, "
            f"sample_rate={sr}Hz, samples={len(y)}"
        )

        # Calculate chunk parameters
        chunk_samples = int(chunk_duration_sec * sr)
        num_chunks = int(np.ceil(total_duration / chunk_duration_sec))

        logger.info(
            f"Chunking audio into {num_chunks} chunks "
            f"({chunk_duration_sec}s each)"
        )

        extracted_chunks = []

        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * chunk_samples
            end_sample = min(start_sample + chunk_samples, len(y))

            # Extract chunk samples
            chunk_audio = y[start_sample:end_sample]

            # Calculate temporal boundaries
            start_sec = (start_sample / sr)
            end_sec = (end_sample / sr)

            # Generate deterministic chunk filename
            chunk_filename = f"{video_id}_chunk_{chunk_idx:03d}.wav"
            chunk_path = os.path.join(output_dir, chunk_filename)

            # Save chunk to file using librosa
            try:
                import soundfile
                soundfile.write(chunk_path, chunk_audio, sr)
            except ImportError:
                # Fallback: use scipy if soundfile not available
                try:
                    from scipy.io import wavfile
                    # Convert to int16 for WAV format
                    chunk_audio_int16 = np.int16(chunk_audio / np.max(np.abs(chunk_audio)) * 32767)
                    wavfile.write(chunk_path, sr, chunk_audio_int16)
                except ImportError:
                    logger.error(
                        "Neither soundfile nor scipy.io.wavfile available. "
                        "Install one with: pip install soundfile or scipy"
                    )
                    raise

            logger.debug(
                f"Saved chunk {chunk_idx}: {chunk_filename} "
                f"({start_sec:.2f}s - {end_sec:.2f}s)"
            )

            # Create chunk metadata
            chunk = ExtractedAudioChunk(
                chunk_id=f"chunk_{chunk_idx:03d}",
                audio_chunk_path=chunk_path,
                spectrogram_image_path="",  # Will be filled after spectrogram generation
                start_sec=round(start_sec, 3),
                end_sec=round(end_sec, 3)
            )
            extracted_chunks.append(chunk)

        logger.info(f"Successfully created {len(extracted_chunks)} audio chunks")
        return extracted_chunks

    except Exception as e:
        logger.error(f"Error during audio chunking: {e}")
        raise RuntimeError(f"Audio chunking error: {str(e)}") from e


def generate_spectrograms(
    audio_chunks: List[ExtractedAudioChunk],
    output_dir: str = TEMP_SPECTROGRAMS_DIR
) -> List[ExtractedAudioChunk]:
    """
    Generate Mel spectrogram images for each audio chunk.

    Mel spectrograms convert audio into visual representation optimized for
    human perception and AI analysis:
    - Mel scale mimics human auditory system (perceptually-motivated)
    - Spectrograms reveal frequency content over time
    - Useful for detecting synthetic audio artifacts (compression, noise)
    - Can be fed to multimodal LLMs for audio-visual analysis
    
    Why Mel spectrograms for synthetic audio detection:
    - Real speech has consistent formants (frequency patterns)
    - Synthetic/processed audio shows compression artifacts
    - Voice conversion systems leave detectable artifacts in spectrogram
    - Frequency distortions are visible in mel-scale representation

    Args:
        audio_chunks: List of ExtractedAudioChunk objects
        output_dir: Directory to save spectrogram images

    Returns:
        Updated list of ExtractedAudioChunk objects with spectrogram paths

    Raises:
        RuntimeError: If dependencies missing or spectrogram generation fails
        ValueError: If audio chunks invalid
    """

    if not HAS_LIBROSA or not HAS_MATPLOTLIB:
        missing = []
        if not HAS_LIBROSA:
            missing.append("librosa")
        if not HAS_MATPLOTLIB:
            missing.append("matplotlib")
        raise RuntimeError(
            f"Required packages missing: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )

    if not audio_chunks:
        raise ValueError("audio_chunks cannot be empty")

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise

    try:
        updated_chunks = []

        for chunk_idx, chunk in enumerate(audio_chunks):
            if not os.path.exists(chunk.audio_chunk_path):
                logger.warning(f"Audio chunk not found: {chunk.audio_chunk_path}")
                continue

            logger.debug(f"Generating spectrogram for {chunk.audio_chunk_path}")

            try:
                # Load audio chunk
                y, sr = librosa.load(chunk.audio_chunk_path, sr=STANDARD_SAMPLE_RATE)

                # Compute Mel spectrogram
                # n_mels: number of mel frequency bins
                # n_fft: FFT size for short-time Fourier transform
                # hop_length: number of samples between successive frames
                S = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    n_mels=MEL_BINS,
                    n_fft=FFT_SIZE,
                    hop_length=HOP_LENGTH
                )

                # Convert to dB scale (log scaling for better visualization)
                S_db = librosa.power_to_db(S, ref=np.max)

                # Create figure and axes (clean layout without extra whitespace)
                fig, ax = plt.subplots(figsize=(10, 4), dpi=SPECTROGRAM_DPI)

                # Display spectrogram
                img = librosa.display.specshow(
                    S_db,
                    sr=sr,
                    hop_length=HOP_LENGTH,
                    x_axis='time',
                    y_axis='mel',
                    ax=ax,
                    cmap='viridis'
                )

                # Configure axes to be minimal and clean
                ax.set_title(f"Mel Spectrogram: {chunk.chunk_id}", fontsize=10)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')

                # Remove extra whitespace
                plt.tight_layout()

                # Generate deterministic spectrogram filename
                spec_filename = f"{chunk.chunk_id}_spectrogram.png"
                spec_path = os.path.join(output_dir, spec_filename)

                # Save figure
                fig.savefig(spec_path, bbox_inches='tight', pad_inches=0.1, dpi=SPECTROGRAM_DPI)
                plt.close(fig)

                logger.debug(f"Saved spectrogram to: {spec_path}")

                # Update chunk with spectrogram path
                updated_chunk = ExtractedAudioChunk(
                    chunk_id=chunk.chunk_id,
                    audio_chunk_path=chunk.audio_chunk_path,
                    spectrogram_image_path=spec_path,
                    start_sec=chunk.start_sec,
                    end_sec=chunk.end_sec
                )
                updated_chunks.append(updated_chunk)

            except Exception as e:
                logger.error(f"Failed to generate spectrogram for chunk {chunk_idx}: {e}")
                # Continue processing other chunks
                continue

        logger.info(f"Generated {len(updated_chunks)} spectrograms")
        return updated_chunks

    except Exception as e:
        logger.error(f"Error during spectrogram generation: {e}")
        raise RuntimeError(f"Spectrogram generation error: {str(e)}") from e


def cleanup_audio(output_dir: str = TEMP_AUDIO_DIR) -> None:
    """
    Clean up temporary extracted audio files.

    Args:
        output_dir: Directory containing temporary audio files

    Raises:
        RuntimeError: If cleanup fails (except for missing directory)
    """
    try:
        if not os.path.exists(output_dir):
            logger.debug(f"Audio directory does not exist: {output_dir}")
            return

        output_path = Path(output_dir)
        audio_files = list(output_path.glob(f"*.{STANDARD_FORMAT}"))

        if not audio_files:
            logger.debug(f"No audio files found in {output_dir}")
            return

        for audio_file in audio_files:
            try:
                os.remove(audio_file)
                logger.debug(f"Deleted: {audio_file}")
            except OSError as e:
                logger.warning(f"Failed to delete {audio_file}: {e}")

        logger.info(f"Cleaned up {len(audio_files)} temporary audio files")

    except Exception as e:
        logger.error(f"Error during audio cleanup: {e}")
        raise RuntimeError(f"Audio cleanup failed: {str(e)}") from e


def cleanup_audio_chunks(output_dir: str = TEMP_AUDIO_CHUNKS_DIR) -> None:
    """
    Clean up temporary audio chunk files.

    Args:
        output_dir: Directory containing temporary audio chunks

    Raises:
        RuntimeError: If cleanup fails (except for missing directory)
    """
    try:
        if not os.path.exists(output_dir):
            logger.debug(f"Audio chunks directory does not exist: {output_dir}")
            return

        output_path = Path(output_dir)
        chunk_files = list(output_path.glob("*.wav"))

        if not chunk_files:
            logger.debug(f"No chunk files found in {output_dir}")
            return

        for chunk_file in chunk_files:
            try:
                os.remove(chunk_file)
                logger.debug(f"Deleted: {chunk_file}")
            except OSError as e:
                logger.warning(f"Failed to delete {chunk_file}: {e}")

        logger.info(f"Cleaned up {len(chunk_files)} temporary audio chunk files")

    except Exception as e:
        logger.error(f"Error during audio chunks cleanup: {e}")
        raise RuntimeError(f"Audio chunks cleanup failed: {str(e)}") from e


def cleanup_spectrograms(output_dir: str = TEMP_SPECTROGRAMS_DIR) -> None:
    """
    Clean up temporary spectrogram image files.

    Args:
        output_dir: Directory containing temporary spectrograms

    Raises:
        RuntimeError: If cleanup fails (except for missing directory)
    """
    try:
        if not os.path.exists(output_dir):
            logger.debug(f"Spectrograms directory does not exist: {output_dir}")
            return

        output_path = Path(output_dir)
        spec_files = list(output_path.glob("*.png"))

        if not spec_files:
            logger.debug(f"No spectrogram files found in {output_dir}")
            return

        for spec_file in spec_files:
            try:
                os.remove(spec_file)
                logger.debug(f"Deleted: {spec_file}")
            except OSError as e:
                logger.warning(f"Failed to delete {spec_file}: {e}")

        logger.info(f"Cleaned up {len(spec_files)} temporary spectrogram files")

    except Exception as e:
        logger.error(f"Error during spectrograms cleanup: {e}")
        raise RuntimeError(f"Spectrograms cleanup failed: {str(e)}") from e


def cleanup_all_audio() -> None:
    """
    Clean up all temporary audio-related directories.
    Convenience function for complete cleanup.
    """
    logger.info("Cleaning up all temporary audio files...")
    cleanup_audio()
    cleanup_audio_chunks()
    cleanup_spectrograms()
    logger.info("Audio cleanup complete")
