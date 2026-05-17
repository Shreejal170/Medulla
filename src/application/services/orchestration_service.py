import asyncio

import json

import logging

import os


from dotenv import load_dotenv

from google import genai


from src.adapters.input.kafka_listener import ConfluentKafkaListenerAdapter

from src.adapters.output.gemini_llm_adapter import GeminiLlmAdapter

from src.adapters.output.kafka_publisher import ConfluentKafkaPublisherAdapter

from src.application.services.frame_analysis_service import FrameAnalysisService

from src.application.services.video_ingestion import VideoIngestionService

from src.core.logging_config import setup_logging

from src.domain.models.analysis import VideoExtractionData


load_dotenv()

setup_logging()

logger = logging.getLogger(__name__)
RESULTS_FILE = "frame_analysis_results.json"


def setup_dependencies():
    """Sets up and returns all necessary dependencies for the orchestration service.

    Args:
        None
    Returns:
        A tuple containing initialized instances of:
        - FrameAnalysisService: The service responsible for analyzing video frames using the LLM.
        - ConfluentKafkaListenerAdapter: The adapter for listening to Kafka topics for incoming frame data.
        - ConfluentKafkaPublisherAdapter: The adapter for publishing analysis results back to Kafka.
        - VideoIngestionService: The service responsible for ingesting videos and extracting frames.
    """

    client = genai.Client(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

    publisher = ConfluentKafkaPublisherAdapter(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    )

    ingestion_service = VideoIngestionService(publisher=publisher)

    listener = ConfluentKafkaListenerAdapter(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
        group_id=os.getenv("KAFKA_GROUP_ID"),
    )

    llm_service = GeminiLlmAdapter(client)

    frame_analysis_service = FrameAnalysisService(llm_service)

    return frame_analysis_service, listener, publisher, ingestion_service


async def orchestrate_analytics():
    """Main orchestration function to set up dependencies, listen for incoming frame data, analyze frames using the LLM, and publish results.
    Args:
        None
    Returns:
        None
    """

    logger.info("Starting Orchestration Service...")

    try:
        frame_analysis_service, listener, publisher, ingestion_service = (
            setup_dependencies()
        )

    except Exception as e:
        logger.error(f"Error in setting up dependencies: {str(e)}", exc_info=True)

        raise

    await publisher.start()

    frames_topic = os.getenv("KAFKA_FRAMES_TOPIC", "frames-ready-for-ai")

    results_topic = os.getenv("KAFKA_RESULTS_TOPIC", "frame-analysis-results")

    async def handler(raw_message: str):

        logger.info("Received message from Kafka, starting analysis...")

        try:
            data = json.loads(raw_message)

            video_extraction_data = VideoExtractionData.model_validate(data)

            analysis_result = await frame_analysis_service.analyze_frame(
                video_extraction_data
            )

            await publisher.publish(topic=results_topic, message=analysis_result)

            try:
                if hasattr(analysis_result, "model_dump"):
                    result_dict = analysis_result.model_dump()

                else:
                    result_dict = analysis_result.dict()

                if os.path.exists(RESULTS_FILE):
                    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                        existing = json.load(f)

                else:
                    existing = []

                existing.append(result_dict)

                with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2)

            except Exception as file_err:
                logger.error(f"Failed to write local JSON: {file_err}", exc_info=True)

            logger.info(
                f"Processed + stored results for video {video_extraction_data.video_id}"
            )

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)

    video_id = os.getenv("TEST_VIDEO_ID", "test_video_001")

    base_path = os.getenv("TEST_VIDEO_PATH", "")

    file_path = os.path.join(base_path, f"{video_id}.mp4") if base_path else ""

    if file_path:
        await ingestion_service.ingest_video(video_id=video_id, file_path=file_path)

        logger.info("Ingestion complete, waiting for frames...")

    else:
        logger.warning("TEST_VIDEO_PATH not set. Skipping ingestion step.")

    listener.subscribe(topic=frames_topic, handler=handler)

    await listener.start()

    logger.info(f"Listening on topic: {frames_topic}")

    try:
        while True:
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        pass

    finally:
        await listener.stop()

        await publisher.stop()
