import logging
import os

from livekit.plugins.openai import llm

from dotenv import load_dotenv
load_dotenv('.env.local')
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, deepgram, silero, turn_detector, google, elevenlabs

def load_instructions():
    with open("law.txt", "r", encoding="utf-8") as file:
        return file.read().strip()

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):

    instructions = load_instructions()

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=instructions #now loads dynamically from corresponding prompt
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    voice_config = elevenlabs.Voice(
        id="EXAVITQu4vr4xnSDxMaL",
        name="Bella",
        category="premade",
        settings=elevenlabs.VoiceSettings(
            stability=0.71,
            similarity_boost=0.5,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    elevenlabs_tts = elevenlabs.TTS(
        voice=voice_config,
        model="eleven_flash_v2_5",
        api_key=os.getenv("ELEVEN_API_KEY"),
        streaming_latency=3,
        enable_ssml_parsing=False,
        chunk_length_schedule=[80, 120, 200, 260],
    )

    cerebras_llm = llm.LLM.with_cerebras(
        model="llama3.1-8b", #adjust if needed
        api_key=os.getenv("CEREBRAS_API_KEY"),
        temperature=0.8,
    )



    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=cerebras_llm,
        tts=elevenlabs_tts,
        turn_detector=turn_detector.EOUModel(),
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
        chat_ctx=initial_ctx,
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    agent.start(ctx.room, participant)

    await agent.say("Hello, how may I assist with your New Jesery Law concerns today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
