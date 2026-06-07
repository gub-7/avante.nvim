from typing import Literal

from pydantic import BaseModel, Field


class GPUDevice(BaseModel):
    vendor: Literal["nvidia", "amd", "intel", "apple", "unknown"]
    name: str
    uuid: str | None = None
    vram_bytes: int | None = None
    free_vram_bytes: int | None = None
    driver: str | None = None
    compute_capability: str | None = None
    gfx_target: str | None = None
    supports_cuda: bool = False
    supports_rocm: bool = False
    supports_vulkan: bool = False


class HardwareProfile(BaseModel):
    os: str
    detected_in: Literal["host", "container", "merged", "unknown"] = "container"
    cpu_model: str | None = None
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_bytes: int = 0
    gpus: list[GPUDevice] = []
    probe_warnings: list[str] = []
    captured_at: str


class BackendRecommendation(BaseModel):
    backend: Literal["ollama", "llama.cpp", "vllm", "sglang", "openai-compatible"]
    accelerator: Literal["cuda", "rocm", "vulkan", "cpu"]
    reason: str
    env: dict[str, str] = {}
    launch_args: list[str] = []
    risk: Literal["low", "medium", "high"] = "low"


class ModelRuntimePlan(BaseModel):
    model_name: str
    quantization: str
    model_bytes: int
    context_tokens: int
    batch_size: int
    expected_concurrent_requests: int
    kv_cache_bytes_estimate: int
    required_vram_bytes: int
    fits_in_vram: bool
    recommendation: str


class OffloadPlan(BaseModel):
    gpu_layers: int | Literal["all"]
    main_gpu: int | None = None
    tensor_split: list[float] | None = None
    split_mode: Literal["none", "layer", "row"] = "none"
    context_size: int
    batch_size: int
    ubatch_size: int | None = None
    kv_cache_type: str | None = None


class BenchmarkResult(BaseModel):
    backend: str
    model: str
    context_tokens: int
    prompt_eval_tps: float
    decode_tps: float
    ttft_ms: float
    peak_vram_bytes: int | None = None
    cpu_percent: float = 0.0
    gpu_percent: float | None = None
    passed: bool = Field(alias="pass", default=True)


class HardwareAwareRagBudget(BaseModel):
    max_retrieved_tokens: int
    max_spans: int
    max_tool_log_tokens: int
    max_agentic_retrieval_steps: int
    reason: str

