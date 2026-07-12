"""
VieNeu-TTS v3 Turbo (int8) — CPU streaming demo (FastAPI).
==========================================================
Stream 48 kHz audio ngay khi generate, qua `V3TurboVieNeuTTS.infer_stream` (đường
ONNX/CPU int8 mặc định). Vì RTF < 1 (int8 nhanh hơn realtime), stream chạy mượt
không underrun — chỉ cần player prebuffer ~300–500ms.

    uv run python -m apps.web_stream        # http://localhost:8001

Public API dùng ở đây:
    tts = Vieneu()                                  # v3 Turbo int8, CPU
    for chunk in tts.infer_stream(text, voice="Trúc Ly"):
        ...                                         # np.float32 @ 48kHz, phát/ghi dần
"""
import time
import io
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel
import uvicorn

from vieneu import Vieneu

SAMPLE_RATE = 48_000
app = FastAPI()
tts = None

ROOT_DIR = Path(__file__).resolve().parents[1]
CLIENT_HTML_PATH = ROOT_DIR / "client" / "client.html"


def load_html_content() -> str:
    if CLIENT_HTML_PATH.exists():
        content = CLIENT_HTML_PATH.read_text(encoding="utf-8")
        if content.lstrip().startswith("HTML_CONTENT = r\"\"\""):
            start = content.find('r"""') + len('r"""')
            end = content.rfind('"""')
            if 0 < start < end:
                return content[start:end]
        return content
    return "<!doctype html><html><body><h1>client.html not found</h1></body></html>"


def load_model():
    global tts
    print("⏳ Loading VieNeu-TTS v3 Turbo (int8, CPU)...")
    # Mặc định: mode=v3turbo, precision=int8, device=auto→cpu. Truyền precision="fp32"
    # để đối chứng chất lượng (chậm hơn trên CPU).
    tts = Vieneu()  # == Vieneu(mode="v3turbo", precision="int8")
    print(f"✅ Ready. Backbone: int8 | intra_op threads: {getattr(tts.engine, 'ort_intra_op_threads', '?')}")


load_model()


@app.get("/")
async def ui():
    return HTMLResponse(load_html_content())


@app.get("/favicon.ico")
async def favicon():
    # SVG emoji favicon → tab hiện 🐆, và hết log 404 favicon.
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y="82" font-size="82">🐆</text></svg>'
    return Response(svg, media_type="image/svg+xml")


@app.get("/voices")
async def voices():
    try:
        vs = tts.list_preset_voices()
        out = []
        for item in vs:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                label, vid = item
                out.append({"id": vid, "name": label})
            else:
                out.append({"id": str(item), "name": str(item)})
        return out or [{"id": "", "name": "(no preset voices)"}]
    except Exception as e:  # noqa: BLE001
        return [{"id": "", "name": f"⚠️ {e}"}]


def _pcm16(audio_f32: np.ndarray) -> bytes:
    return (np.asarray(audio_f32) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


@app.get("/stream")
async def stream(text: str, voice_id: Optional[str] = None):
    """Stream 48 kHz WAV: header (nframes rất lớn để phát liên tục) rồi PCM16 theo chunk."""
    def gen():
        # WAV header 48 kHz mono; nframes huge → browser phát liền khi data tới.
        h = io.BytesIO()
        with wave.open(h, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.setnframes(1_000_000_000)
        yield h.getvalue()

        t0 = time.perf_counter()
        first_at = None
        n_chunks = 0
        emitted = 0
        for chunk in tts.infer_stream(text, voice=voice_id or None):
            if chunk is None or len(chunk) == 0:
                continue
            if first_at is None:
                first_at = time.perf_counter() - t0
                print(f"⚡ TTFA (time-to-first-audio): {first_at*1000:.0f} ms")
            n_chunks += 1
            emitted += len(chunk)
            yield _pcm16(chunk)
        if first_at is not None:
            gen_time = time.perf_counter() - t0
            audio_s = emitted / SAMPLE_RATE
            rtf = gen_time / audio_s if audio_s else 0
            print(f"✅ {n_chunks} chunks | audio {audio_s:.2f}s | gen {gen_time:.2f}s "
                  f"| RTF {rtf:.3f} ({1/rtf:.1f}x realtime)" if rtf else "")

    return StreamingResponse(gen(), media_type="audio/wav")


class StreamReq(BaseModel):
    text: str
    voice_id: Optional[str] = None


@app.post("/stream")
async def stream_post(req: StreamReq):
    return await stream(req.text, req.voice_id)


def main():
    print("🌍 Mở http://localhost:8001 để test VieNeu v3 Turbo (int8) streaming (CPU)")
    uvicorn.run(app, host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
