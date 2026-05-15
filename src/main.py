import asyncio
import logging
import json
from src.core.logging_config import setup_logging
from src.domain.models.analysis import VideoExtractionData
from google import genai
# from src.application.services.orchestration_service import orchestrate_analytics
from src.application.services.frame_analysis_service import FrameAnalysisService
from src.application.services.video_ingestion import VideoExtractionAndIngestionService
from src.adapters.output.kafka_publisher import ConfluentKafkaPublisherAdapter
from src.adapters.input.kafka_listener import ConfluentKafkaListenerAdapter
from src.adapters.output.gemini_llm_adapter import GeminiLlmAdapter
from src.application.services.decision_engine import run_decision_engine
from src.core.llm_config import llm_settings


setup_logging()
logger = logging.getLogger("main.py")



async def get_publisher():
    publisher = ConfluentKafkaPublisherAdapter(bootstrap_servers='localhost:9092')
    # await publisher.start()
    return publisher

async def get_boostrap_server():
    boostrap_server = "localhost:9092"
    return boostrap_server

async def get_group_id():
    group_id = 'fovea-ai-workers'
    return group_id

async def get_videos_file_path():
    return r"./data/input_videos/video.mp4"

async def get_topic():
    return "frames-ready-for-ai"

async def get_ai_adapter():
    client = genai.Client(api_key=llm_settings.google_gemini_api_key)
    gemini_adapter = GeminiLlmAdapter(client)
    return gemini_adapter

async def store_result_json(data):
    try:
        with open('./data/result/final_result.json', 'w') as f:
            logger.info("Saving into the json file.")
            data = json.loads(data)
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(e)


async def generate_decision(frame_analysis_result):
    try:
        logger.info("Generating Final Decision")
        final_decision = run_decision_engine(frame_analysis_result)
        return final_decision
    except Exception as e:
        logger.error(e)

## CC
async def frame_analysis(video_extraction_data):
    frame_analysis_service = FrameAnalysisService(await get_ai_adapter())
    result = await frame_analysis_service.analyze_frame(video_extraction_data)
    return result

## DD
async def custom_handler(raw_string):
    logger.info(raw_string)
    logger.info("New message...")
    try:
        video_extraction_data = json.loads(raw_string)
        if VideoExtractionData.model_validate(video_extraction_data):
            logger.info("Data Validated")
            frame_analysis_result = await frame_analysis(video_extraction_data)
            final_decision = await generate_decision(frame_analysis_result)
            await store_result_json(final_decision)
        else:
            logger.error("Data type is not matching.")
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON.")
    except Exception as e:
        logger.error(f"Data type is not matching or processing failed: {e}")


async def orchestrate_analytics():

    #1. Consumer
    video_consumer = ConfluentKafkaListenerAdapter(await get_boostrap_server(), await get_group_id())
    video_consumer.subscribe(await get_topic(), custom_handler)
    await video_consumer.start()
    logger.info("Consumer is listening. Press Ctrl+C to stop.")
    logger.info("Waiting 3 seconds for Kafka to assign partitions...")
    await asyncio.sleep(3)


    #2.  Publisher
    publisher = await get_publisher()

    #2. Video injector instance
    video_ingester_publisher = VideoExtractionAndIngestionService(publisher=publisher)

    # Extracted and published to kafka
    try:
        await publisher.start()
        await video_ingester_publisher.ingest_video("Vid123", await get_videos_file_path())
    except Exception as e:
        logger.error(e)
    finally:
        await publisher.stop()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await video_consumer.stop()


if __name__ == "__main__":
    asyncio.run(orchestrate_analytics())


