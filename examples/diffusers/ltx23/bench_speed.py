import os, sys, time, asyncio, logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.pool import SubprocessPool

OUT = os.environ.get("BENCH_OUT", "/tmp/bench-out")
os.makedirs(OUT, exist_ok=True)
W, H, NF, FPS, STEPS, GS, SEED = 1920, 1088, 121, 24, 5, 1.0, 42
SHAPE = "%dx%d@%df" % (W, H, NF)
PROMPTS = [
    ("musician", "A street musician in her thirties sings and strums an acoustic guitar on a sunny city sidewalk, natural human face, photorealistic, sharp focus."),
    ("dog", "A close-up tracking shot of a golden retriever sprinting through a sunlit alpine meadow at golden hour, photorealistic"),
]


def req(tag, prompt):
    return {
        "request_id": "bench_" + tag, "prompt": prompt, "width": W, "height": H,
        "num_frames": NF, "fps": FPS, "num_inference_steps": STEPS,
        "guidance_scale": GS, "seed": SEED, "negative_prompt": None,
        "output_path": os.path.join(OUT, tag + ".mp4"),
    }


async def main():
    pool = SubprocessPool(
        model_path="/data/default", num_gpus=1, enable_optimizations=True,
        attention_backend="TORCH_SDPA",
        model_factory_dotted="ltx23.factory:load_model", model_label="ltx2-3-distilled",
    )
    try:
        t = time.perf_counter()
        r = await pool.route(SHAPE, req("warm", "a calm test warmup clip"))
        print("BOOT_WARM_S=%.1f status=%s" % (time.perf_counter() - t, r.get("status")), flush=True)
        for tag, prompt in PROMPTS:
            t = time.perf_counter()
            r = await pool.route(SHAPE, req(tag, prompt))
            print("GEN[%s]=%.1fs status=%s" % (tag, time.perf_counter() - t, r.get("status")), flush=True)
        print("BENCH_ALL_DONE", flush=True)
    finally:
        await pool.shutdown()


asyncio.run(main())
