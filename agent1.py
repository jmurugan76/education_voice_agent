from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import openai, noise_cancellation, silero

# Load environment variables
load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful realtime voice assistant. "
                "Listen carefully, respond briefly and clearly, "
                "and avoid emojis or fancy formatting."
            )
        )


# Create Agent Server
server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        turn_detection="vad",
        stt=openai.STT(model="gpt-4o-transcribe"),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice="ash",
            instructions="Speak in a friendly, natural tone."
        ),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                )
            )
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and explain you are a realtime LiveKit Python voice agent."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
# CLI: python agent.py console | dev | start