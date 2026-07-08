"""Bake + bench driver for the LTX-2.3 SPEED image (see RUNBOOK.md).

  python3 ltx23/bake_bench.py t2v   # pass 1: t2v compile + Mega-Cache blob + warm bench
  python3 ltx23/bake_bench.py i2v   # pass 2: extend the blob with i2v graphs
                                    # (move the pass-1 blob aside; LTX_MEGACACHE_SAVE_EVERY=1)

Exits non-zero if a warm generation exceeds MAX_WARM_S: a silently-slowed bake
(wrong preset, package drift, upstream change) must fail here, not reach prod.
"""
import asyncio, logging, os, sys, time

logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.pool import SubprocessPool
from ltx23.config import profile_attention_backend

OUT = os.environ.get("BENCH_OUT", "/tmp/bench-out")
os.makedirs(OUT, exist_ok=True)
W, H, NF, FPS, STEPS, GS = 1920, 1088, 121, 24, 5, 1.0
SHAPE = "%dx%d@%df" % (W, H, NF)
# Derive from the profile (same source as ltx23/worker.py + warmup.py) instead
# of hardcoding a backend -- a hardcoded value silently drifts from whatever
# the served profile actually uses, baking the WRONG kernels into Mega-Cache.
PROFILE = os.environ.get("LTX23_PROFILE", "speed")
ATTENTION_BACKEND = os.environ.get(
    "FASTVIDEO_ATTENTION_BACKEND", profile_attention_backend(PROFILE)
)
MAX_WARM_S = float(os.environ.get("LTX23_MAX_WARM_S", "10"))
I2V_IMAGE = os.environ.get("LTX23_I2V_IMAGE", "/work/joy_nordic_woman_mid.png")
T2V_PROMPTS = [
    ("musician", "A street musician in her thirties sings and strums an acoustic guitar on a sunny city sidewalk, natural human face, photorealistic, sharp focus."),
    ("dog", "A close-up tracking shot of a golden retriever sprinting through a sunlit alpine meadow at golden hour, photorealistic"),
]


def req(tag: str, prompt: str, **extra) -> dict:
    d = {
        "request_id": "bake_" + tag, "prompt": prompt, "width": W, "height": H,
        "num_frames": NF, "fps": FPS, "num_inference_steps": STEPS,
        "guidance_scale": GS, "seed": 42, "negative_prompt": None,
        "output_path": os.path.join(OUT, tag + ".mp4"),
    }
    d.update(extra)
    return d


async def main(mode: str) -> int:
    pool = SubprocessPool(
        model_path=os.environ.get("LTX23_MODEL_PATH", "/models/ltx-2.3-distilled-diffusers"),
        num_gpus=1, enable_optimizations=True, attention_backend=ATTENTION_BACKEND,
        model_factory_dotted="ltx23.factory:load_model", model_label="ltx23-bake",
    )
    warm_times: list[tuple[str, float]] = []
    try:
        t = time.perf_counter()
        r = await pool.route(SHAPE, req("warm", "a calm test warmup clip"))
        print("BOOT_WARM_S=%.1f status=%s" % (time.perf_counter() - t, r.get("status")), flush=True)
        if mode == "t2v":
            gens = [(tag, req(tag, prompt)) for tag, prompt in T2V_PROMPTS]
        else:
            gens = [("i2v", req(
                "i2v", "The scene comes alive with gentle natural motion and a slow cinematic push-in.",
                ltx2_images=[(I2V_IMAGE, 0, 1.0)], ltx2_image_crf=0.0))]
        for tag, r_ in gens:
            t = time.perf_counter()
            r = await pool.route(SHAPE, r_)
            dt = time.perf_counter() - t
            warm_times.append((tag, dt))
            print("GEN[%s]=%.1fs status=%s" % (tag, dt, r.get("status")), flush=True)
        print("BAKE_BENCH_DONE mode=%s" % mode, flush=True)
    finally:
        await pool.shutdown()
    # Latency gate only applies to the t2v pass: there the first gen is the cold
    # compile and the rest (warm_times[1:]) are genuinely warm cache hits. The
    # i2v pass has a single gen that is ALWAYS a cold compile (it exists to
    # extend the Mega-Cache blob with i2v graphs), so there is no warm sample to
    # gate -- skip it there to avoid a guaranteed false-positive failure.
    if mode == "t2v":
        slow = [(tag, dt) for tag, dt in warm_times[1:] if dt > MAX_WARM_S]
        if slow:
            print("LATENCY GATE FAILED (> %.0fs warm): %s -- do NOT ship this bake; "
                  "check the preset boot log and recent env/package changes." % (MAX_WARM_S, slow), flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "t2v")))
