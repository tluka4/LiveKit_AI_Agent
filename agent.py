import logging
import os

from livekit.plugins.openai import llm
from livekit.plugins.cartesia import tts


from dotenv import load_dotenv
load_dotenv('.env.local')
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    llm as agents_llm
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

    initial_ctx = agents_llm.ChatContext().append(
        role="system",
        text=instructions #now loads dynamically from corresponding prompt
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    cartesia_tts = tts.TTS(
        model="sonic-turbo",
        voice="c2ac25f9-ecc4-4f56-9095-651354df60c0",  # "Codey"
        speed=0.4,
        emotion=["curiosity:highest", "positivity:high"]
    )


    cerebras_llm = llm.LLM.with_cerebras(
        model="llama3.1-70b", #adjust if needed
        temperature=0.8,
    )



    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=cerebras_llm,
        tts=cartesia_tts,
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

    await agent.say("How may I address your New Jersey Law concerns?", allow_interruptions=True)
    #Welcome to XYZ real estate, how may i assist you?


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )

