# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM worker initialization for TensorRT-LLM backend.

This module handles the initialization and lifecycle of text and multimodal
LLM workers using TensorRT-LLM.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Optional

from prometheus_client import REGISTRY
from tensorrt_llm.llmapi import (
    CapacitySchedulerPolicy,
    DisaggregatedParams,
    DynamicBatchConfig,
    KvCacheConfig,
    SchedulerConfig,
)
from tensorrt_llm.llmapi.llm import SamplingParams
from tensorrt_llm.llmapi.llm_args import (
    TOKENIZER_ALIASES,
    KvCacheConnectorConfig,
    LoadFormat,
)
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory
from tensorrt_llm.metrics import MetricsCollector
from torch.cuda import device_count
from transformers import AutoConfig

import dynamo.nixl_connect as nixl_connect
from dynamo import prometheus_names
from dynamo.common.config_dump import dump_config
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.prometheus import (
    LLMBackendMetrics,
    register_embedding_cache_metrics,
    register_engine_metrics_callback,
)
from dynamo.common.utils.runtime import parse_endpoint
from dynamo.common.utils.topology import apply_topology_config
from dynamo.llm import (
    KvEventPublisher,
    MediaDecoder,
    MediaFetcher,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime
from dynamo.trtllm.args import Config
from dynamo.trtllm.constants import DisaggregationMode, Modality
from dynamo.trtllm.engine import Backend, get_llm_engine
from dynamo.trtllm.health_check import (
    TrtllmHealthCheckPayload,
    _get_bos_token_id_from_tokenizer,
)
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor
from dynamo.trtllm.publisher import DYNAMO_COMPONENT_REGISTRY, get_publisher
from dynamo.trtllm.request_handlers.handlers import (
    RequestHandlerConfig,
    RequestHandlerFactory,
)
from dynamo.trtllm.utils.trtllm_utils import deep_update, get_spec_decode_runtime_data

# Default buffer size for kv cache events.
DEFAULT_KV_EVENT_BUFFER_MAX_SIZE = 100_000
SPEC_DECODE_RUNTIME_KEY = "spec_decode"


def build_kv_connector_config(config: Config):
    if config.connector:
        if config.connector[0] == "kvbm":
            return KvCacheConnectorConfig(
                connector_module="kvbm.trtllm_integration.connector",
                connector_scheduler_class="DynamoKVBMConnectorLeader",
                connector_worker_class="DynamoKVBMConnectorWorker",
            )
        elif config.connector[0] == "none":
            return None
        else:
            logging.error(f"Invalid connector: {config.connector[0]}")
            sys.exit(1)
    return None


def _warn_override_collisions(target: dict, source: dict, path: str = "") -> None:
    """Log warnings for keys in *source* that will overwrite existing values in *target*."""
    for key, new_val in source.items():
        full_key = f"{path}.{key}" if path else key
        if key in target:
            old_val = target[key]
            if isinstance(new_val, dict) and isinstance(old_val, dict):
                _warn_override_collisions(old_val, new_val, full_key)
            elif old_val != new_val:
                logging.warning(
                    "override_engine_args will replace %s: %r -> %r",
                    full_key,
                    old_val,
                    new_val,
                )


def _parse_model_loader_extra_config(raw: object) -> dict[str, object]:
    """Parse --model-loader-extra-config into a dict. Accepts a dict or a JSON string."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in --model-loader-extra-config: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("--model-loader-extra-config must decode to a JSON object")
        return parsed
    raise ValueError(
        "--model-loader-extra-config must be a JSON object string or a dict"
    )


def _sync_config_from_engine_args(config: Config, engine_args: dict) -> None:
    """Sync MDC-visible config fields from final TensorRT-LLM engine args."""
    for field_name in ("max_seq_len", "max_num_tokens", "max_batch_size"):
        if field_name in engine_args:
            setattr(config, field_name, engine_args[field_name])


def _register_memory_routes(runtime, handler) -> None:
    # DEEPINFRA: MM (deepapi EvictDrainedShards) drives drain via /engine/sleep
    # and /engine/wake_up — vLLM-native names. Register aliases so the drain
    # protocol works on TRT-LLM workers, and register unconditionally (the GMS
    # gate at the call sites is dropped) so non-GMS deployments get the routes.
    runtime.register_engine_route("sleep", handler.release_memory_occupation)
    runtime.register_engine_route("wake_up", handler.resume_memory_occupation)
    runtime.register_engine_route(
        "control/release_memory_occupation",
        handler.release_memory_occupation,
    )
    runtime.register_engine_route(
        "control/resume_memory_occupation",
        handler.resume_memory_occupation,
    )
    logging.info(
        "Registered engine routes: /engine/sleep, /engine/wake_up, "
        "/engine/control/release_memory_occupation, "
        "/engine/control/resume_memory_occupation"
    )


# Generous bound for the 1-token capacity probe: engine init (including CUDA
# graph capture) has already completed by the time it runs, so the real cost
# is one prefill+decode iteration.
_KV_CAPACITY_PROBE_GENERATION_TIMEOUT_S = 180
_KV_CAPACITY_PROBE_STATS_TIMEOUT_S = 30


def _reset_iteration_result_singletons(engine) -> None:
    """Drop TRT-LLM's lazily-created IterationResult singletons after the probe.

    ``aget_stats`` constructs ``_iter_stats_result`` (and the sibling
    ``_iter_kv_events_result``) on first use; its AsyncQueue's asyncio.Event
    binds to the event loop that first *awaits* it — during the probe, the
    main loop. The Publisher's stats thread (ManagedThread) runs a private
    loop and awaits the same singleton, which then raises "bound to a
    different event loop" and kills the stats thread: frozen trtllm_* and
    dynamo_component_* gauges, no kv_used_blocks ActiveLoad, no FPM.
    Dropping the singletons restores the pre-probe lazy-init order — the
    publisher's first poll rebuilds them on its own loop.
    """
    executor = getattr(engine.llm, "_executor", None)
    if executor is None:
        return
    for attr in ("_iter_stats_result", "_iter_kv_events_result"):
        if hasattr(executor, attr):
            setattr(executor, attr, None)


async def _probe_total_kv_blocks(engine, tokenizer, config: Config) -> Optional[int]:
    """Learn the per-rank KV block capacity (``kvCacheStats.maxNumBlocks``).

    TRT-LLM only exposes the KV pool size through iteration stats, and the
    engine enqueues those only when a request actually iterates. Run a single
    1-token probe generation — the same shape as the health-check canary —
    and then drain the stats queue to read ``maxNumBlocks``.

    This runs before ``register_model``, so no external traffic can race the
    stats queue (the publisher's stats thread also starts later). The frontend
    needs ``total_kv_blocks`` in the runtime config for
    ``--active-decode-blocks-threshold`` overload rejection: without it the
    worker monitor has no denominator and the threshold is a silent no-op.

    Returns None (never raises) if the capacity cannot be determined;
    registration then proceeds without ``total_kv_blocks``, matching the old
    behavior.
    """
    if config.disaggregation_mode == DisaggregationMode.ENCODE:
        # MultimodalEncoder: no KV cache, no generate surface.
        return None
    if not config.publish_events_and_metrics:
        # enable_iter_perf_stats is tied to this flag; without it the engine
        # never enqueues kvCacheStats, so there is nothing to probe.
        logging.warning(
            "Cannot probe total_kv_blocks: iteration stats are disabled "
            "(publish_events_and_metrics=False). "
            "--active-decode-blocks-threshold will not take effect for this worker."
        )
        return None
    if os.getenv("DYN_TRTLLM_SKIP_KV_CAPACITY_PROBE") == "1":
        logging.warning(
            "Skipping total_kv_blocks probe (DYN_TRTLLM_SKIP_KV_CAPACITY_PROBE=1)."
        )
        return None

    bos_token_id = _get_bos_token_id_from_tokenizer(tokenizer)
    # end_id/pad_id must be set explicitly: the engine may run with
    # skip_tokenizer_init, in which case TRT-LLM cannot infer them. Any valid
    # token id works — max_tokens=1 ends the request regardless.
    sampling_params = SamplingParams(
        max_tokens=1,
        end_id=bos_token_id,
        pad_id=bos_token_id,
        detokenize=False,
    )
    disaggregated_params = None
    if config.disaggregation_mode in (
        DisaggregationMode.PREFILL,
        DisaggregationMode.DECODE,
    ):
        # Same trick as TrtllmHealthCheckPayload: run the probe as a local
        # prefill+decode so no KV transceiver or peer worker is required.
        disaggregated_params = DisaggregatedParams(
            request_type="context_and_generation"
        )

    try:
        try:
            generation = engine.llm.generate_async(
                inputs=[bos_token_id],
                sampling_params=sampling_params,
                disaggregated_params=disaggregated_params,
                streaming=False,
            )
            await asyncio.wait_for(
                generation.aresult(), timeout=_KV_CAPACITY_PROBE_GENERATION_TIMEOUT_S
            )
        except Exception:
            logging.warning(
                "total_kv_blocks probe generation failed; registering without it.",
                exc_info=True,
            )
            return None

        # The engine serializes iteration stats after the iteration completes;
        # poll briefly. Take the max across rows: under attention DP each rank
        # reports its own (homogeneous) pool, so any row carries the per-rank
        # capacity the worker monitor expects.
        total_kv_blocks = 0
        deadline = time.monotonic() + _KV_CAPACITY_PROBE_STATS_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                async for stat in engine.llm.get_stats_async(timeout=1):
                    if isinstance(stat, str):
                        stat = json.loads(stat)
                    max_blocks = int(
                        (stat.get("kvCacheStats") or {}).get("maxNumBlocks") or 0
                    )
                    total_kv_blocks = max(total_kv_blocks, max_blocks)
            except (asyncio.TimeoutError, TimeoutError, asyncio.QueueEmpty):
                pass
            except Exception:
                logging.warning(
                    "total_kv_blocks probe failed reading engine stats.",
                    exc_info=True,
                )
                return None
            if total_kv_blocks > 0:
                return total_kv_blocks
            await asyncio.sleep(0.5)

        logging.warning(
            "total_kv_blocks probe timed out waiting for kvCacheStats; "
            "registering without it."
        )
        return None
    finally:
        # Must run on every exit path once the engine was touched, or the
        # Publisher's stats thread dies on its first poll (see helper doc).
        _reset_iteration_result_singletons(engine)


async def init_llm_worker(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: Optional[list] = None,
    engine_holder: Optional[list] = None,
) -> None:
    """Initialize and run the LLM worker.

    This function handles text and multimodal LLM modalities using TensorRT-LLM.

    Args:
        runtime: The Dynamo distributed runtime.
        config: Configuration parsed from command line.
        shutdown_event: Event to signal shutdown.
        shutdown_endpoints: Optional list to populate with endpoints for graceful shutdown.
        engine_holder: Optional mutable list; when provided, the TensorRTLLMEngine
            is appended so that the drain callback can reference it at shutdown time.
    """

    encode_client = None
    if config.encode_endpoint:
        logging.info(
            f"Initializing encode worker client for endpoint: {config.encode_endpoint}"
        )
        parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
            config.encode_endpoint
        )
        encode_client = await runtime.endpoint(
            f"{parsed_namespace}.{parsed_component_name}.{parsed_endpoint_name}"
        ).client()

    # Convert model path to Path object if it's a local path, otherwise keep as string
    model_path = str(config.model)

    if config.gpus_per_node is None:
        gpus_per_node = device_count()
        if gpus_per_node == 0:
            raise ValueError("No GPU devices found on the node")
    else:
        gpus_per_node = config.gpus_per_node

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=config.free_gpu_memory_fraction
    )

    if config.has_connector("kvbm"):
        kv_cache_config.enable_partial_reuse = False

    dynamic_batch_config = DynamicBatchConfig(
        enable_batch_size_tuning=True,
        enable_max_num_tokens_tuning=False,
        dynamic_batch_moving_average_window=128,
    )
    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        dynamic_batch_config=dynamic_batch_config,
    )
    kv_connector_config = build_kv_connector_config(config)

    try:
        model_loader_extra_config = _parse_model_loader_extra_config(
            config.model_loader_extra_config
        )
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    if config.load_format == "gms":
        try:
            from gpu_memory_service.integrations.trtllm import setup_gms
        except ImportError as exc:
            raise RuntimeError(
                "gpu-memory-service is required for --load-format gms. "
                "Install or update the package."
            ) from exc
        setup_gms(model_loader_extra_config)
        logging.info(
            "TRT-LLM GMS integration enabled (extra=%s)", model_loader_extra_config
        )

    # Resolve load_format for engine args. GMS patches are active regardless;
    # fall back to "auto" if TRT-LLM doesn't recognise "gms" as a LoadFormat.
    engine_load_format = config.load_format
    if config.load_format == "gms":
        try:
            LoadFormat(config.load_format)
        except (ValueError, KeyError):
            logging.warning(
                "TensorRT-LLM does not recognise load_format='gms'; "
                "using 'auto' while GMS patches remain active."
            )
            engine_load_format = "auto"

    arg_map = {
        "model": model_path,
        "scheduler_config": scheduler_config,
        "tensor_parallel_size": config.tensor_parallel_size,
        "pipeline_parallel_size": config.pipeline_parallel_size,
        "moe_expert_parallel_size": config.expert_parallel_size,
        "enable_attention_dp": config.enable_attention_dp,
        "backend": Backend.PYTORCH,
        "kv_cache_config": kv_cache_config,
        "gpus_per_node": gpus_per_node,
        "max_num_tokens": config.max_num_tokens,
        "max_seq_len": config.max_seq_len,
        "max_beam_width": config.max_beam_width,
        "max_batch_size": config.max_batch_size,
        "return_perf_metrics": config.publish_events_and_metrics,
        # enable_iter_perf_stats is required for PyTorch backend to compute iteration-level
        # stats (KV cache utilization, hit rate). TensorRT backend always has this enabled.
        # See TRT-LLM PR #11243: MetricsCollector.log_iteration_stats() needs these stats.
        "enable_iter_perf_stats": config.publish_events_and_metrics,
        "kv_connector_config": kv_connector_config,
    }

    arg_map["load_format"] = engine_load_format

    # Enable sleep_config when GMS manages weights — required for GMS
    # unmap/remap. Conditional because SleepConfig contains unpicklable
    # lambdas that break MPI-based multi-rank distribution.
    if config.load_format == "gms":
        from tensorrt_llm.llmapi.llm_args import SleepConfig

        arg_map["sleep_config"] = SleepConfig()

    # Add guided decoding backend if specified
    if config.guided_decoding_backend is not None:
        arg_map["guided_decoding_backend"] = config.guided_decoding_backend
        logging.info(
            "Guided decoding enabled with backend: %s",
            config.guided_decoding_backend,
        )

    if config.extra_engine_args != "":
        # TODO: Support extra engine args from json file as well.
        arg_map = update_llm_args_with_extra_options(arg_map, config.extra_engine_args)

    # Apply override_engine_args if provided
    if config.override_engine_args != "":
        try:
            overrides = json.loads(config.override_engine_args)
            logging.info(f"Applying engine arg overrides: {overrides}")

            _warn_override_collisions(arg_map, overrides)
            deep_update(arg_map, overrides)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse override_engine_args as JSON: {e}")
            sys.exit(1)

    _sync_config_from_engine_args(config, arg_map)

    event_buffer_max_size = 0
    if config.publish_events_and_metrics:
        # 'event_buffer_max_size' is required to enable TRTLLM to publish kv cache events.
        # Add it to kv_cache_config while preserving all settings from YAML
        current_kv_config = arg_map["kv_cache_config"]
        if isinstance(current_kv_config, KvCacheConfig):
            # Convert KvCacheConfig object to dict, preserving ALL existing settings
            # This ensures YAML overrides are not lost when adding event_buffer_max_size
            current_kv_config = current_kv_config.model_dump(exclude_none=True)
            arg_map["kv_cache_config"] = current_kv_config

        if not isinstance(current_kv_config, dict):
            raise TypeError(
                "kv_cache_config must be a dict or KvCacheConfig, "
                f"got {type(current_kv_config).__name__}"
            )

        # Preserve a user-specified event_buffer_max_size from YAML/overrides;
        # only apply the default when it is unset or zero (TRTLLM's disabled value).
        existing = current_kv_config.get("event_buffer_max_size")
        if existing:
            logging.info(
                f"Using existing event_buffer_max_size={existing} from kv_cache_config"
            )
        else:
            current_kv_config[
                "event_buffer_max_size"
            ] = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
        event_buffer_max_size = int(current_kv_config["event_buffer_max_size"])

        # Only pytorch backend is supported for now to publish events and metrics.
        if "backend" not in arg_map:
            arg_map["backend"] = Backend.PYTORCH
        elif arg_map["backend"] not in Backend:
            logging.error(
                "Only %s supported for now to publish events and metrics. Got: %s",
                [b.value for b in Backend],
                arg_map["backend"],
            )
            sys.exit(1)

    trtllm_zmq_bind_endpoint = None  # Endpoint for TensorRT-LLM to bind and publish
    consolidator_output_endpoint = (
        None  # Endpoint where consolidator publishes (workers subscribe to this)
    )

    try:
        from kvbm.trtllm_integration.consolidator_config import (
            get_consolidator_endpoints,
            should_enable_consolidator,
        )

        if should_enable_consolidator(arg_map):
            # get_consolidator_endpoints returns (trtllm_bind_endpoint, output_bind_endpoint, output_connect_endpoint)
            consolidator_endpoints = get_consolidator_endpoints()
            trtllm_zmq_bind_endpoint = consolidator_endpoints[0]  # TRTLLM bind endpoint
            consolidator_output_endpoint = consolidator_endpoints[
                1
            ]  # Consolidator output bind endpoint (for KVBM connector)
            consolidator_output_connect_endpoint = consolidator_endpoints[
                2
            ]  # Consolidator output connect endpoint (for worker publisher)
    except ImportError:
        # kvbm package is not installed
        logging.info(
            "kvbm package not installed - skipping KV event consolidator setup."
        )
    except Exception as e:
        logging.error(
            f"Failed to set up consolidator endpoints: {e}. "
            "Continuing without KV event consolidation.",
            exc_info=True,
        )

    logging.info(f"TensorRT-LLM engine args: {arg_map}")
    engine_args = arg_map

    # Populate default sampling params from the model
    custom_tokenizer = arg_map.get("custom_tokenizer")
    if custom_tokenizer:
        from importlib import import_module

        try:
            tokenizer_path = TOKENIZER_ALIASES.get(custom_tokenizer, custom_tokenizer)
            module_path, class_name = tokenizer_path.rsplit(".", 1)
            tokenizer_class = getattr(import_module(module_path), class_name)
            tokenizer = tokenizer_class.from_pretrained(
                arg_map.get("tokenizer") or arg_map["model"],
                trust_remote_code=arg_map.get("trust_remote_code", False),
            )
        except (ValueError, ImportError, AttributeError) as e:
            raise ValueError(
                f"Failed to load custom tokenizer '{custom_tokenizer}': {e}. "
                "Expected format: 'module.path.ClassName' or a recognized alias in TensorRT-LLM LLM API."
            ) from e
    else:
        tokenizer = tokenizer_factory(
            arg_map["model"], trust_remote_code=arg_map.get("trust_remote_code", False)
        )
    default_sampling_params = SamplingParams()

    # Enable perf metrics so prompt_tokens_details can be returned
    if hasattr(default_sampling_params, "return_perf_metrics"):
        default_sampling_params.return_perf_metrics = True
    model_input = ModelInput.Tokens

    # Set model type based on disaggregation mode. Prefill and encode workers
    # carry no OpenAI surface — their role is declared via `worker_type`.
    if config.disaggregation_mode == DisaggregationMode.PREFILL:
        # Prefill registers the legacy `ModelType.Prefill` marker bit (not a
        # surface) so an OLD frontend, which detects prefill via that bit,
        # still routes disaggregated traffic during the cross-version rollout. A new
        # frontend ignores it and dispatches off `worker_type`.
        model_type = ModelType.Prefill
    elif config.disaggregation_mode == DisaggregationMode.ENCODE:
        # Encode helpers expose no surface and (unlike prefill) had no legacy
        # marker bit, so they stay Empty.
        model_type = ModelType.Empty
    else:
        model_type = parse_endpoint_types(config.endpoint_types)
        logging.info(f"Registering model with endpoint types: {config.endpoint_types}")

        # Warn if custom template provided but chat endpoint not enabled
        if config.custom_jinja_template and "chat" not in config.endpoint_types:
            logging.warning(
                "Custom Jinja template provided (--custom-jinja-template) but 'chat' not in --endpoint-types. "
                "The chat template will be loaded but the /v1/chat/completions endpoint will not be available."
            )

    multimodal_processor = None

    if os.getenv("DYN_ENABLE_TEST_LOGITS_PROCESSOR") == "1":
        # We need to initialize the tokenizer for the test logits processor
        # But detokenizing still happens in the rust engine, so we do _not_ want
        # to set default_sampling_params.detokenize to True.
        # This overrides the skip_tokenizer_init=True set earlier
        engine_args["skip_tokenizer_init"] = False

    if config.modality == Modality.MULTIMODAL:
        engine_args["skip_tokenizer_init"] = False
        model_config = AutoConfig.from_pretrained(
            config.model,
            trust_remote_code=engine_args.get("trust_remote_code", False),
        )
        multimodal_processor = MultimodalRequestProcessor(
            model_type=model_config.model_type,
            model_dir=config.model,
            max_file_size_mb=config.max_file_size_mb,
            tokenizer=tokenizer,
            allowed_local_media_path=config.allowed_local_media_path,
            enable_frontend_decoding=config.frontend_decoding,
        )

    else:
        # We already detokenize inside HandlerBase. No need to also do it in TRTLLM.
        default_sampling_params.detokenize = False

    connector = None
    needs_nixl = (
        config.modality == Modality.MULTIMODAL
        and config.disaggregation_mode != DisaggregationMode.AGGREGATED
        and (
            config.frontend_decoding
            or config.disaggregation_mode == DisaggregationMode.ENCODE
            or (
                config.disaggregation_mode == DisaggregationMode.PREFILL
                and bool(config.encode_endpoint)
            )
        )
    )
    if needs_nixl:
        try:
            logging.info("Initializing NIXL Connect.")
            connector = nixl_connect.Connector()
            await connector._create_connection()
        except Exception:
            logging.warning(
                "Failed to initialize NIXL Connect; "
                "KV-cache transfer will be unavailable.",
                exc_info=True,
            )
            connector = None
    else:
        logging.info("Skipping NIXL Connect initialization (aggregated mode).")

    dump_config(
        config.dump_config_to, {"engine_args": engine_args, "dynamo_args": config}
    )

    # Prepare model name for metrics
    model_name_for_metrics = config.served_model_name or config.model

    # Construct Prometheus gauges directly; passed through to the engine and publisher
    # via explicit parameters (no module-level global).
    component_gauges = LLMBackendMetrics(
        registry=DYNAMO_COMPONENT_REGISTRY,
        model_name=model_name_for_metrics,
        component_name=config.component,
    )

    async with get_llm_engine(
        engine_args,
        config.disaggregation_mode,
        component_gauges=component_gauges,
    ) as engine:
        # Expose engine to the drain callback installed by main.py.
        # The callback uses this to poll active request count during shutdown.
        if engine_holder is not None:
            engine_holder.append(engine)

        # Snapshot mode must capture the initialized TRT-LLM/CUDA state before
        # Dynamo runtime endpoints, health routes, or discovery sockets exist.
        # The snapshot runtime proxy waits here for capture/restore and creates
        # the real runtime only after restore; normal runtimes skip this hook.
        snapshot_before_endpoint = getattr(runtime, "snapshot_before_endpoint", None)
        if snapshot_before_endpoint is not None:
            await snapshot_before_endpoint(engine, config)

        engine.start_health_monitor(runtime=runtime, shutdown_event=shutdown_event)

        endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.{config.endpoint}"
        )

        if shutdown_endpoints is not None:
            shutdown_endpoints[:] = [endpoint]

        runtime_config = ModelRuntimeConfig()
        runtime_config.context_length = config.max_seq_len

        # TRT-LLM reports KV capacity only via iteration stats, which need a
        # request to flow. Probe it with a 1-token generation before
        # registration; see _probe_total_kv_blocks. Without total_kv_blocks
        # the frontend's --active-decode-blocks-threshold gate never fires.
        total_kv_blocks = await _probe_total_kv_blocks(engine, tokenizer, config)
        if total_kv_blocks:
            runtime_config.total_kv_blocks = total_kv_blocks

        # Set values from config that are available immediately
        # Note: We populate max_num_seqs and max_num_batched_tokens from config
        # to ensure Prometheus metrics are available even without engine stats

        # Naming clarification:
        # - In vLLM: max_num_seqs = maximum concurrent requests (this is an unusual name due to vLLM's historic reasons)
        # - In TensorRT-LLM: max_batch_size = maximum concurrent requests (clearer name)
        # Both parameters control the same thing: how many requests can be processed simultaneously

        # Need to get max_num_seqs and max_num_batched_tokens from engine_args
        # because they can be overridden by --extra-engine-args or --override-engine-args
        runtime_config.max_num_seqs = engine_args["max_batch_size"]
        runtime_config.max_num_batched_tokens = engine_args["max_num_tokens"]
        runtime_config.reasoning_parser = config.dyn_reasoning_parser
        runtime_config.tool_call_parser = config.dyn_tool_call_parser
        runtime_config.exclude_tools_when_tool_choice_none = (
            config.exclude_tools_when_tool_choice_none
        )
        runtime_config.set_structural_tag_mode(
            "on" if config.dyn_enable_structural_tag else "off"
        )
        runtime_config.set_structural_tag_scope(config.dyn_structural_tag_scope)
        runtime_config.set_structural_tag_schema(config.dyn_structural_tag_schema)
        # Decode workers don't create the WorkerKvQuery endpoint, so don't advertise local indexer
        runtime_config.enable_local_indexer = (
            config.enable_local_indexer
            and config.disaggregation_mode != DisaggregationMode.DECODE
        )
        # Set data_parallel_size for attention DP mode
        # This enables the router's scheduler to correctly iterate over all dp_ranks
        # Need to name ADP as `data_parallel_size` for parity with other frameworks
        attention_dp_size = engine.get_attention_dp_size()
        runtime_config.data_parallel_size = attention_dp_size

        # Set topology and KV transfer policy for topology-aware routing
        apply_topology_config(runtime_config)

        spec_decode_runtime_data = get_spec_decode_runtime_data(engine_args)
        if spec_decode_runtime_data is not None:
            runtime_config.set_engine_specific(
                SPEC_DECODE_RUNTIME_KEY,
                json.dumps(spec_decode_runtime_data),
            )
            logging.info(
                "Published TRT-LLM spec decode runtime metadata: %s",
                spec_decode_runtime_data,
            )

        logging.info(f"Set runtime config max_num_seqs: {runtime_config.max_num_seqs}")
        logging.info(
            f"Set runtime config max_num_batched_tokens: {runtime_config.max_num_batched_tokens}"
        )
        logging.info(f"Set runtime config data_parallel_size: {attention_dp_size}")
        logging.info(
            f"Set runtime config total_kv_blocks: {runtime_config.total_kv_blocks} "
            "(per-rank, probed from engine kvCacheStats.maxNumBlocks)"
        )

        # Initialize TensorRT-LLM MetricsCollector and register with global REGISTRY
        # This enables exposing TRT-LLM's native Prometheus metrics (request latency, TTFT, TPOT, etc.)
        metrics_collector = None
        additional_metrics = None
        if config.publish_events_and_metrics:
            try:
                model_name_for_metrics = config.served_model_name or config.model
                metrics_collector = MetricsCollector(
                    {"model_name": model_name_for_metrics, "engine_type": "trtllm"}
                )
                logging.info("TensorRT-LLM MetricsCollector initialized")

                # Prefix filter: all TRT-LLM metrics (engine + additional) use "trtllm_" prefix
                _metric_prefixes = ["trtllm_"]

                # Additional metrics (abort tracking, request types, KV transfer perf).
                # Wrapped in try/except because AdditionalMetricsCollector depends on
                # prometheus_names which may not be available in all packaging variants.
                try:
                    from dynamo.trtllm.metrics import AdditionalMetricsCollector

                    disagg_mode_str = (
                        config.disaggregation_mode.value
                        if hasattr(config.disaggregation_mode, "value")
                        else str(config.disaggregation_mode)
                    )
                    additional_metrics = AdditionalMetricsCollector(
                        labels={
                            "model_name": model_name_for_metrics,
                            "disaggregation_mode": disagg_mode_str,
                            "engine_type": "trtllm",
                        },
                    )
                    logging.info(
                        "Additional metrics initialized (disagg_mode=%s)",
                        disagg_mode_str,
                    )
                except Exception as e:
                    logging.warning("Failed to initialize additional metrics: %s", e)

                # Single callback for all Python-side metrics (trtllm_ + additional)
                register_engine_metrics_callback(
                    endpoint=endpoint,
                    registry=REGISTRY,
                    metric_prefix_filters=_metric_prefixes,
                    namespace_name=config.namespace,
                    component_name=config.component,
                    endpoint_name="generate",
                    model_name=model_name_for_metrics,
                )
                logging.info(
                    "Prometheus metrics registered (prefixes: %s)", _metric_prefixes
                )
            except Exception as e:
                logging.warning(
                    f"Failed to initialize TensorRT-LLM Prometheus metrics: {e}"
                )

        # Register callback for Dynamo component metrics using dedicated registry
        register_engine_metrics_callback(
            endpoint=endpoint,
            registry=DYNAMO_COMPONENT_REGISTRY,
        )
        logging.debug("DYNAMO_COMPONENT_REGISTRY callback registered successfully")

        # publisher will be set later if publishing is enabled.
        handler_config = RequestHandlerConfig(
            engine=engine,
            default_sampling_params=default_sampling_params,
            publisher=None,
            disaggregation_mode=config.disaggregation_mode,
            encode_client=encode_client,
            multimodal_processor=multimodal_processor,
            generate_endpoint=endpoint,
            connector=connector,
            runtime=runtime,  # Pass runtime for graceful shutdown
            metrics_collector=metrics_collector,
            kv_block_size=config.kv_block_size,
            shutdown_event=shutdown_event,
            encoder_cache_capacity_gb=config.multimodal_embedding_cache_capacity_gb,
            additional_metrics=additional_metrics,
            max_seq_len=config.max_seq_len,
            disagg_machine_id=int(endpoint.connection_id()) % 1021,
        )

        media_decoder = None
        media_fetcher = None
        if config.frontend_decoding:
            media_decoder = MediaDecoder()
            media_decoder.enable_image({"limits": {"max_alloc": 128 * 1024 * 1024}})
            media_fetcher = MediaFetcher()
            media_fetcher.timeout_ms(30000)
            allow_internal = os.getenv("DYN_MM_ALLOW_INTERNAL", "0") == "1"
            media_fetcher.allow_direct_ip(allow_internal)
            media_fetcher.allow_direct_port(allow_internal)

        # Register the model with runtime config for every disaggregation
        # role, including ENCODE. Encode workers get their own bucket in the
        # WorkerSet via `worker_type` in the ws_key.
        if config.disaggregation_mode == DisaggregationMode.PREFILL:
            worker_type = WorkerType.Prefill
            needs_set: list[WorkerType] = [WorkerType.Decode]
        elif config.disaggregation_mode == DisaggregationMode.DECODE:
            worker_type = WorkerType.Decode
            needs_set = [WorkerType.Prefill]
        elif config.disaggregation_mode == DisaggregationMode.ENCODE:
            worker_type = WorkerType.Encode
            # Encode workers want either a P+D pair, or a single Aggregated
            # peer that handles both stages. DNF: outer OR, inner AND.
            needs_set = []  # placeholder, overridden below
        else:
            # AGGREGATED ("prefill_and_decode")
            worker_type = WorkerType.Aggregated
            needs_set = []
        # `--encode-endpoint` is non-empty when this worker talks to a
        # separate encode worker; that adds Encode to its needs.
        if worker_type != WorkerType.Encode and getattr(
            config, "encode_endpoint", None
        ):
            needs_set.append(WorkerType.Encode)
        if worker_type == WorkerType.Encode:
            needs: list[list[WorkerType]] = [
                [WorkerType.Prefill, WorkerType.Decode],
                [WorkerType.Aggregated],
            ]
        else:
            needs = [needs_set] if needs_set else []

        await register_model(
            model_input,
            model_type,
            endpoint,
            config.model,
            config.served_model_name,
            kv_cache_block_size=config.kv_block_size,
            runtime_config=runtime_config,
            custom_template_path=config.custom_jinja_template,
            media_decoder=media_decoder,
            media_fetcher=media_fetcher,
            worker_type=worker_type,
            needs=needs,
        )

        health_check_payload = TrtllmHealthCheckPayload(
            tokenizer=tokenizer,
            disaggregation_mode=config.disaggregation_mode,
        ).to_dict()

        if config.publish_events_and_metrics:
            # Initialize and pass in the publisher to the request handler to
            # publish events and metrics.
            # Use model as fallback if served_model_name is not provided
            model_name_for_metrics = config.served_model_name or config.model
            metrics_labels = [
                (
                    prometheus_names.labels.MODEL,
                    model_name_for_metrics,
                ),  # OpenAI standard
                (
                    prometheus_names.labels.MODEL_NAME,
                    model_name_for_metrics,
                ),  # Native engine compatibility
            ]

            # Create worker-side publisher for consolidated events if consolidator is enabled
            # This subscribes to consolidator's ZMQ output and publishes to NATS with worker_id
            consolidator_publisher = None
            if consolidator_output_endpoint:
                # Use the connect endpoint directly (already provided by get_consolidator_endpoints)
                consolidator_publisher = KvEventPublisher(
                    endpoint=endpoint,
                    kv_block_size=config.kv_block_size,
                    zmq_endpoint=consolidator_output_connect_endpoint,
                    zmq_topic="",
                    enable_local_indexer=config.enable_local_indexer,
                )
                logging.info(
                    f"Created worker-side publisher for consolidated events: "
                    f"subscribing to {consolidator_output_connect_endpoint}, worker_id={endpoint.connection_id()}"
                )

            async with get_publisher(
                endpoint,
                engine,
                int(endpoint.connection_id()),
                config.kv_block_size,
                metrics_labels,
                component_gauges=component_gauges,
                additional_metrics=additional_metrics,
                event_buffer_max_size=event_buffer_max_size,
                zmq_endpoint=trtllm_zmq_bind_endpoint,
                enable_local_indexer=config.enable_local_indexer,
                metrics_collector=metrics_collector,
            ) as publisher:
                handler_config.publisher = publisher
                handler = RequestHandlerFactory().get_request_handler(handler_config)
                # DEEPINFRA: register drain routes unconditionally (was gated on
                # load_format=="gms"; MM needs /engine/sleep on all deployments).
                _register_memory_routes(runtime, handler)

                encoder_cache = getattr(handler, "_encoder_cache", None)
                if encoder_cache is not None:
                    register_embedding_cache_metrics(
                        endpoint=endpoint,
                        cache=encoder_cache,
                        model_name=model_name_for_metrics,
                        component_name=config.component,
                    )
                await endpoint.serve_endpoint(
                    handler.generate,
                    metrics_labels=metrics_labels,
                    health_check_payload=health_check_payload,
                )

            # Shutdown consolidator publisher if it was created
            if consolidator_publisher:
                consolidator_publisher.shutdown()
        else:
            handler = RequestHandlerFactory().get_request_handler(handler_config)
            # DEEPINFRA: register drain routes unconditionally (see above).
            _register_memory_routes(runtime, handler)
            await endpoint.serve_endpoint(
                handler.generate, health_check_payload=health_check_payload
            )
