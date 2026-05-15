#!/usr/bin/env python3
"""
AUDIO PIPELINE USAGE EXAMPLES

Copy these code snippets to integrate audio extraction into your workflows.
All examples assume audio_helper is installed and dependencies are available.
"""

# ============================================================================
# EXAMPLE 1: Basic Local Audio Extraction (Standalone)
# ============================================================================

def example_basic_extraction():
    """Extract audio and spectrograms from a local video file."""
    from src.utils.audio_helper import (
        extract_audio,
        chunk_audio,
        generate_spectrograms,
        cleanup_all_audio
    )
    
    video_path = "path/to/video.mp4"
    video_id = "sample_video"
    
    try:
        # Step 1: Extract audio
        print("Extracting audio...")
        audio_path = extract_audio(video_id, video_path)
        print(f"✓ Audio extracted: {audio_path}")
        
        # Step 2: Chunk audio
        print("Chunking audio into segments...")
        chunks = chunk_audio(audio_path, video_id, chunk_duration_sec=5.0)
        print(f"✓ Created {len(chunks)} audio chunks")
        
        # Step 3: Generate spectrograms
        print("Generating spectrograms...")
        chunks_with_specs = generate_spectrograms(chunks)
        print(f"✓ Generated {len(chunks_with_specs)} spectrograms")
        
        # Display results
        for chunk in chunks_with_specs:
            print(f"\n{chunk.chunk_id}:")
            print(f"  Audio: {chunk.audio_chunk_path}")
            print(f"  Spectrogram: {chunk.spectrogram_image_path}")
            print(f"  Time: {chunk.start_sec:.1f}s - {chunk.end_sec:.1f}s")
        
        # Optional cleanup
        # cleanup_all_audio()
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True


# ============================================================================
# EXAMPLE 2: Complete Pipeline with AudioExtractionData
# ============================================================================

def example_complete_pipeline_with_model():
    """Extract audio and return structured AudioExtractionData."""
    from src.utils.audio_helper import (
        extract_audio,
        chunk_audio,
        generate_spectrograms
    )
    from src.domain.models.analysis import AudioExtractionData
    import json
    
    video_path = "path/to/video.mp4"
    video_id = "complete_example"
    
    try:
        # Run pipeline
        audio_path = extract_audio(video_id, video_path)
        chunks = chunk_audio(audio_path, video_id)
        chunks_with_specs = generate_spectrograms(chunks)
        
        # Create structured data
        extraction_data = AudioExtractionData(
            video_id=video_id,
            extracted_audio_chunks=chunks_with_specs
        )
        
        # Output as JSON
        output_json = extraction_data.model_dump_json(indent=2)
        print(output_json)
        
        # Save to file
        with open(f"audio_extraction_{video_id}.json", "w") as f:
            f.write(output_json)
        
        return extraction_data
        
    except Exception as e:
        print(f"Error: {e}")
        return None


# ============================================================================
# EXAMPLE 3: Convert Spectrograms to Base64 for LLM
# ============================================================================

def example_spectrogram_to_base64():
    """Convert spectrogram images to base64 for multimodal LLM."""
    import base64
    import json
    from src.utils.audio_helper import extract_audio, chunk_audio, generate_spectrograms
    from src.domain.models.analysis import AudioExtractionData
    
    video_path = "path/to/video.mp4"
    video_id = "llm_example"
    
    try:
        # Run pipeline
        audio_path = extract_audio(video_id, video_path)
        chunks = chunk_audio(audio_path, video_id)
        chunks_with_specs = generate_spectrograms(chunks)
        
        extraction_data = AudioExtractionData(
            video_id=video_id,
            extracted_audio_chunks=chunks_with_specs
        )
        
        # Convert to base64
        llm_payloads = []
        
        for chunk in extraction_data.extracted_audio_chunks:
            # Read spectrogram image
            with open(chunk.spectrogram_image_path, 'rb') as f:
                spec_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Create LLM payload
            payload = {
                "video_id": extraction_data.video_id,
                "chunk_id": chunk.chunk_id,
                "start_sec": chunk.start_sec,
                "end_sec": chunk.end_sec,
                "spectrogram_base64": spec_base64,
                # Include this for reference (can load raw audio if needed)
                "audio_chunk_path": chunk.audio_chunk_path
            }
            llm_payloads.append(payload)
        
        # Save payloads for LLM processing
        with open(f"llm_payloads_{video_id}.json", "w") as f:
            json.dump(llm_payloads, f, indent=2)
        
        print(f"✓ Created {len(llm_payloads)} LLM payloads")
        
        return llm_payloads
        
    except Exception as e:
        print(f"Error: {e}")
        return None


# ============================================================================
# EXAMPLE 4: Combine Frame and Audio Analysis
# ============================================================================

def example_combined_frame_audio_analysis():
    """Process both frame and audio, correlate by timestamp."""
    from src.utils.audio_helper import extract_audio, chunk_audio, generate_spectrograms
    from src.utils.ffmpeg_helper import extract_frames
    from src.domain.models.ingestion import VideoIngestionEvent
    
    video_path = "path/to/video.mp4"
    video_id = "combined_example"
    
    try:
        # Create ingestion event
        event = VideoIngestionEvent(
            video_id=video_id,
            file_path=video_path,
            sampling_fps=0.5
        )
        
        # Extract frames
        print("Extracting frames...")
        frame_data = extract_frames(
            video_id=event.video_id,
            file_path=event.file_path,
            sampling_fps=event.sampling_fps
        )
        print(f"✓ Extracted {frame_data.total_frames} frames")
        
        # Extract audio
        print("Extracting audio...")
        audio_path = extract_audio(event.video_id, event.file_path)
        chunks = chunk_audio(audio_path, event.video_id)
        audio_data = generate_spectrograms(chunks)
        print(f"✓ Extracted {len(audio_data)} audio chunks")
        
        # Correlate by timestamp
        print("\nCorrelating frame and audio by timestamp...")
        for frame in frame_data.extracted_frames:
            frame_time = frame.timestamp_sec
            
            # Find audio chunks that overlap with this frame
            matching_chunks = [
                c for c in audio_data
                if c.start_sec <= frame_time < c.end_sec
            ]
            
            if matching_chunks:
                print(f"Frame {frame.frame_id} ({frame_time:.2f}s):")
                print(f"  Frame: {frame.frame_file_path}")
                for chunk in matching_chunks:
                    print(f"  Audio Chunk: {chunk.chunk_id} ({chunk.start_sec:.1f}s-{chunk.end_sec:.1f}s)")
                    print(f"    Spectrogram: {chunk.spectrogram_image_path}")
        
        return frame_data, audio_data
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None


# ============================================================================
# EXAMPLE 5: Batch Processing Multiple Videos
# ============================================================================

def example_batch_process_videos():
    """Process multiple video files in sequence."""
    import os
    from pathlib import Path
    from src.utils.audio_helper import (
        extract_audio,
        chunk_audio,
        generate_spectrograms
    )
    from src.domain.models.analysis import AudioExtractionData
    
    video_directory = "path/to/videos"
    results = []
    
    # Find all video files
    video_files = list(Path(video_directory).glob("*.mp4"))
    print(f"Found {len(video_files)} videos")
    
    for idx, video_path in enumerate(video_files, 1):
        try:
            print(f"\n[{idx}/{len(video_files)}] Processing: {video_path.name}")
            
            video_id = video_path.stem  # Filename without extension
            
            # Run pipeline
            audio_path = extract_audio(str(video_id), str(video_path))
            chunks = chunk_audio(audio_path, video_id)
            chunks_with_specs = generate_spectrograms(chunks)
            
            # Create extraction data
            extraction_data = AudioExtractionData(
                video_id=video_id,
                extracted_audio_chunks=chunks_with_specs
            )
            
            results.append({
                "video_path": str(video_path),
                "video_id": video_id,
                "total_chunks": extraction_data.total_audio_chunks,
                "extraction_data": extraction_data
            })
            
            print(f"✓ Extracted {extraction_data.total_audio_chunks} audio chunks")
            
        except Exception as e:
            print(f"✗ Error processing {video_path}: {e}")
            continue
    
    print(f"\n✓ Successfully processed {len(results)} videos")
    return results


# ============================================================================
# EXAMPLE 6: Custom Chunk Duration for Different Use Cases
# ============================================================================

def example_custom_chunk_durations():
    """Demonstrate different chunk durations for different scenarios."""
    from src.utils.audio_helper import extract_audio, chunk_audio, generate_spectrograms
    from src.domain.models.analysis import AudioExtractionData
    
    video_path = "path/to/video.mp4"
    video_id = "custom_chunks"
    
    # Different chunk durations for different scenarios
    chunk_configs = {
        "short": 2.0,      # For detailed analysis, more granular
        "medium": 5.0,     # Default, balanced
        "long": 10.0,      # For broader context
    }
    
    all_results = {}
    
    for config_name, chunk_duration in chunk_configs.items():
        try:
            print(f"\nProcessing with {config_name} chunks ({chunk_duration}s)...")
            
            # Extract once, chunk differently
            audio_path = extract_audio(video_id, video_path)
            chunks = chunk_audio(
                audio_path,
                video_id,
                chunk_duration_sec=chunk_duration,
                output_dir=f"temp_audio_chunks_{config_name}"
            )
            chunks_with_specs = generate_spectrograms(chunks)
            
            extraction_data = AudioExtractionData(
                video_id=f"{video_id}_{config_name}",
                extracted_audio_chunks=chunks_with_specs
            )
            
            all_results[config_name] = extraction_data
            print(f"✓ Created {extraction_data.total_audio_chunks} chunks")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return all_results


# ============================================================================
# EXAMPLE 7: With Explicit Error Handling and Logging
# ============================================================================

def example_with_logging():
    """Production-ready example with comprehensive logging."""
    import logging
    from src.utils.audio_helper import (
        extract_audio,
        chunk_audio,
        generate_spectrograms,
        cleanup_all_audio
    )
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    video_path = "path/to/video.mp4"
    video_id = "logging_example"
    
    try:
        logger.info(f"Starting audio extraction for {video_path}")
        
        # Extract audio
        logger.debug(f"Extracting audio from {video_path}")
        audio_path = extract_audio(video_id, video_path)
        logger.info(f"Audio extracted successfully: {audio_path}")
        
        # Chunk audio
        logger.debug(f"Chunking audio into 5-second segments")
        chunks = chunk_audio(audio_path, video_id)
        logger.info(f"Audio chunked into {len(chunks)} segments")
        
        # Generate spectrograms
        logger.debug(f"Generating Mel spectrograms for {len(chunks)} chunks")
        chunks_with_specs = generate_spectrograms(chunks)
        logger.info(f"Spectrograms generated successfully")
        
        # Summary
        logger.info(f"Audio extraction complete")
        logger.info(f"Results available in:")
        logger.info(f"  temp_audio/{video_id}.wav")
        logger.info(f"  temp_audio_chunks/")
        logger.info(f"  temp_spectrograms/")
        
        # Cleanup (optional)
        logger.debug(f"Cleaning up temporary files")
        cleanup_all_audio()
        logger.info(f"Cleanup complete")
        
        return chunks_with_specs
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


# ============================================================================
# EXAMPLE 8: Integration with Kafka Producer (Future Pipeline)
# ============================================================================

def example_kafka_integration():
    """
    Example showing how to integrate with Kafka pipeline.
    Uncomment and configure for your Kafka setup.
    """
    # from confluent_kafka import Producer
    from src.utils.audio_helper import extract_audio, chunk_audio, generate_spectrograms
    from src.domain.models.analysis import AudioExtractionData
    import json
    
    video_path = "path/to/video.mp4"
    video_id = "kafka_example"
    
    # Configure Kafka (example - customize for your setup)
    # kafka_config = {
    #     'bootstrap.servers': 'localhost:9092',
    #     'client.id': 'audio-producer'
    # }
    # producer = Producer(kafka_config)
    
    try:
        # Run audio extraction pipeline
        audio_path = extract_audio(video_id, video_path)
        chunks = chunk_audio(audio_path, video_id)
        chunks_with_specs = generate_spectrograms(chunks)
        
        extraction_data = AudioExtractionData(
            video_id=video_id,
            extracted_audio_chunks=chunks_with_specs
        )
        
        # Send each chunk to Kafka
        for chunk in extraction_data.extracted_audio_chunks:
            message = {
                "video_id": extraction_data.video_id,
                "chunk_id": chunk.chunk_id,
                "audio_chunk_path": chunk.audio_chunk_path,
                "spectrogram_image_path": chunk.spectrogram_image_path,
                "start_sec": chunk.start_sec,
                "end_sec": chunk.end_sec
            }
            
            # Produce to Kafka
            # producer.produce(
            #     topic='audio-chunks-topic',
            #     key=f"{video_id}_{chunk.chunk_id}".encode(),
            #     value=json.dumps(message).encode(),
            #     callback=delivery_report
            # )
            
            print(f"Would send: {message}")
        
        # Flush Kafka producer
        # producer.flush()
        
        # Cleanup only after Kafka confirms
        # from src.utils.audio_helper import cleanup_all_audio
        # cleanup_all_audio()
        
        return extraction_data
        
    except Exception as e:
        print(f"Error: {e}")
        return None


# ============================================================================
# RUN EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Basic extraction", example_basic_extraction),
        "2": ("Complete pipeline with model", example_complete_pipeline_with_model),
        "3": ("Spectrogram to base64", example_spectrogram_to_base64),
        "4": ("Combined frame + audio", example_combined_frame_audio_analysis),
        "5": ("Batch processing", example_batch_process_videos),
        "6": ("Custom chunk durations", example_custom_chunk_durations),
        "7": ("With logging", example_with_logging),
        "8": ("Kafka integration (reference)", example_kafka_integration),
    }
    
    print("Audio Pipeline Usage Examples")
    print("=" * 50)
    print("\nAvailable examples:")
    for num, (name, _) in examples.items():
        print(f"  {num}. {name}")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            print(f"\nRunning: {examples[choice][0]}")
            print("-" * 50)
            examples[choice][1]()
        else:
            print(f"Invalid choice: {choice}")
    else:
        print("\nUsage: python examples.py <number>")
        print("Example: python examples.py 1")
