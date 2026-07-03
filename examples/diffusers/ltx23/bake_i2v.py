"""Bake helper: one t2v + one i2v through the pool so the Mega-Cache blob
covers both modes. Run on a warm per-shape cache with the old blob moved
aside (a present blob suppresses saving) and LTX_MEGACACHE_SAVE_EVERY=1."""
import asyncio, sys, time
sys.path.insert(0, "/opt/app")
from lib.pool import SubprocessPool

PROMPT = ("A street musician playing saxophone under warm evening light, "
          "cinematic, photorealistic.")
IMG = "/work/joy_nordic_woman_mid.png"

async def main() -> None:
    pool = SubprocessPool(
        model_path="/models/ltx-2.3-distilled-diffusers",
        model_label="ltx23-bake",
        num_gpus=1,
        enable_optimizations=True,
        attention_backend="TORCH_SDPA",
        model_factory_dotted="ltx23.factory:load_model",
    )
    base = dict(height=1088, width=1920, num_frames=121, fps=24,
                num_inference_steps=5, guidance_scale=1.0,
                negative_prompt="", save_video=True)
    t0 = time.perf_counter()
    await pool.generate("1920x1088@121f", request_id="bake-t2v",
                        prompt=PROMPT, seed=7,
                        output_path="/tmp/bench-out/bake_t2v.mp4", **base)
    print(f"T2V_S={time.perf_counter()-t0:.1f}", flush=True)
    t0 = time.perf_counter()
    await pool.generate("1920x1088@121f", request_id="bake-i2v",
                        prompt="The scene comes alive with gentle natural "
                               "motion and a slow cinematic push-in.",
                        seed=7, ltx2_images=[(IMG, 0, 1.0)], ltx2_image_crf=0.0,
                        output_path="/tmp/bench-out/bake_i2v.mp4", **base)
    print(f"I2V_S={time.perf_counter()-t0:.1f}", flush=True)
    await pool.shutdown()
    print("BAKE_I2V_DONE", flush=True)

asyncio.run(main())
