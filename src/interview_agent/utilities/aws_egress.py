import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from livekit import api
from livekit.agents import (
    AgentSession,
    JobContext,
)

logger = logging.getLogger("interview-agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
load_dotenv(".env.local")


def generate_session_folder_path(
    user_details: dict[str, Any], readable_timestamp: str
) -> str:
    """
    Generate consistent folder structure for session files.

    Args:
        user_details: Dictionary containing user information
        room_name: LiveKit room name

    Returns:
        Folder path in format: {user_name}/{readable_timestamp}/{room_name}
        Example: Aaraj/2024-12-15-14-30-52/
        The trailing slash is for S3 compatibility.
    """
    user_name = user_details.get("name", "unknown_user")
    return f"{user_name}/{readable_timestamp}/"


def create_s3_config() -> api.S3Upload:
    """Create S3 upload configuration from environment variables."""
    return api.S3Upload(
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region=os.getenv("AWS_REGION", "ap-southeast-2"),
        bucket=os.getenv("AWS_S3_BUCKET", "refobe-document-analytics"),
    )


def create_egress_request(
    room_name: str, user_details: dict[str, Any], readable_timestamp: str
) -> api.RoomCompositeEgressRequest:
    """Create egress request for room recording."""
    s3_config = create_s3_config()
    folder_path = generate_session_folder_path(user_details, readable_timestamp)
    return api.RoomCompositeEgressRequest(
        room_name=room_name,
        layout="speaker",
        preset=api.EncodingOptionsPreset.H264_720P_30,
        audio_only=False,
        segment_outputs=[
            api.SegmentedFileOutput(
                filename_prefix=folder_path,
                playlist_name="playlist.m3u8",
                live_playlist_name="live_playlist.m3u8",
                segment_duration=6,
                s3=s3_config,
            )
        ],
    )


async def setup_egress_recording(
    ctx: JobContext, user_details: dict[str, Any], readable_timestamp: str
) -> str:
    """Setup S3 egress recording and return egress ID."""
    egress_request = create_egress_request(
        ctx.room.name, user_details, readable_timestamp
    )
    folder_path = generate_session_folder_path(user_details, readable_timestamp)

    # Initialize LiveKit API
    lk_api = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )

    try:
        egress_response = await lk_api.egress.start_room_composite_egress(
            egress_request
        )
        logger.info(f"Egress started successfully: {egress_response.egress_id}")
        logger.info(f"Recording will be saved to S3 folder: {folder_path}")
        return egress_response.egress_id
    except Exception as e:
        logger.error(f"Failed to start S3 egress: {e}")
        return None


async def setup_transcript_callback(
    ctx: JobContext,
    session: AgentSession,
    user_details: dict[str, Any],
    readable_timestamp: str,
) -> None:
    """Setup transcript saving callback to save both locally and to S3."""

    async def write_transcript():
        folder_path = generate_session_folder_path(user_details, readable_timestamp)

        # Create local directory structure
        local_dir = os.path.join("/home/azureuser/voice-agent", folder_path)

        # Ensure the folder path doesn't end with a slash
        folder_path = folder_path.rstrip("/")

        # Create local filename
        local_filename = os.path.join(local_dir, "transcript.json")

        # Create S3 key using the exact folder_path provided
        s3_key = f"{folder_path}/transcript.json"

        try:
            transcript_data = session.history.to_dict()
            os.makedirs(local_dir, exist_ok=True)

            # Save locally
            with open(local_filename, "w") as f:
                json.dump(transcript_data, f, indent=2)
            logger.info(f"Transcript saved locally to {local_filename}")

            # Upload to S3 using thread pool to avoid blocking
            def upload_to_s3():
                try:
                    s3_client = boto3.client(
                        "s3",
                        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                        region_name=os.getenv("AWS_REGION", "ap-southeast-2"),
                    )
                    bucket_name = os.getenv(
                        "AWS_S3_BUCKET", "refobe-document-analytics"
                    )
                    # Upload the file to S3 using the exact folder_path
                    s3_client.upload_file(local_filename, bucket_name, s3_key)
                    logger.info(
                        f"Transcript uploaded to S3: s3://{bucket_name}/{s3_key}"
                    )
                except ClientError as e:
                    logger.error(f"S3 upload error: {e}")
                except Exception as e:
                    logger.error(f"Unexpected upload error: {e}")

            # Execute synchronous upload in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(ThreadPoolExecutor(), upload_to_s3)

        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")

    ctx.add_shutdown_callback(write_transcript)


async def setup_egress_cleanup(ctx: JobContext, egress_id: str) -> None:
    """Setup egress cleanup callback."""

    async def cleanup_egress():
        if egress_id:
            try:
                lk_api = api.LiveKitAPI(
                    url=os.getenv("LIVEKIT_URL"),
                    api_key=os.getenv("LIVEKIT_API_KEY"),
                    api_secret=os.getenv("LIVEKIT_API_SECRET"),
                )
                await lk_api.egress.stop_egress(egress_id)
                logger.info(f"Egress {egress_id} stopped")

                # Close the LiveKit API client if it has cleanup methods
                if hasattr(lk_api, "aclose"):
                    await lk_api.aclose()
                elif hasattr(lk_api, "close"):
                    await lk_api.close()

            except Exception as e:
                logger.warning(f"Failed to stop egress: {e}")

    ctx.add_shutdown_callback(cleanup_egress)
