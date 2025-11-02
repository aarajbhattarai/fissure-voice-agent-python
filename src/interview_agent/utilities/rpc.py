import json
import logging
from typing import Any

from livekit import rtc

logger = logging.getLogger("structured-data-streamer")


class StructuredDataStreamer:
    """Handles streaming structured data (JSON) to frontend via LiveKit."""

    def __init__(self, room: rtc.Room):
        self.room = room
        self.local_participant = room.local_participant
        self._active_streams = {}

    async def send_structured_data(self, topic: str, data: dict[str, Any]) -> None:
        """
        Send structured data as JSON to the frontend.

        Args:
            topic: The topic/channel to send data on
            data: Dictionary payload to send

        """
        try:
            json_str = json.dumps(data, ensure_ascii=False)

            # For single-shot sends, use sendText (opens and closes automatically)
            await self.local_participant.send_text(text=json_str, topic=topic)

            logger.debug(
                f"Sent structured data on topic '{topic}': {len(json_str)} bytes"
            )

        except Exception as e:
            logger.error(
                f"Failed to send structured data on topic '{topic}': {e}", exc_info=True
            )

    async def stream_structured_data(
        self, topic: str, data_chunks: list[dict[str, Any]]
    ) -> None:
        """
        Stream multiple structured data chunks incrementally.

        Args:
            topic: The topic/channel to send data on
            data_chunks: List of dictionary payloads to send incrementally
        """
        writer = None
        try:
            # Open a text stream
            writer = await self.local_participant.stream_text(topic=topic)
            logger.info(f"Opened structured data stream on topic '{topic}' with ID:")

            # Send each chunk as JSON
            for chunk in data_chunks:
                json_str = json.dumps(chunk, ensure_ascii=False)
                await writer.write(json_str)
                print("Hello0------------------------------------ -------  -", json_str)
                logger.debug(f"Wrote chunk to stream: {len(json_str)} bytes")

            # IMPORTANT: Use aclose() not close()
            await writer.aclose()
            logger.info("Closed structured data stream with ID")

        except Exception as e:
            logger.error(
                f"Error streaming structured data on topic '{topic}': {e}",
                exc_info=True,
            )
            if writer:
                try:
                    await writer.aclose()
                except Exception as close_error:
                    logger.error(f"Error closing stream after exception: {close_error}")
