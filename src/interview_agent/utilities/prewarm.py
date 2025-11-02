from livekit.agents import (
    JobProcess,
)
from livekit.plugins import silero


def prewarm(proc: JobProcess) -> None:
    """Prewarm function to load VAD model."""
    proc.userdata["vad"] = silero.VAD.load()
