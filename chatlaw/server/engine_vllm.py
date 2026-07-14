"""基于 vLLM AsyncLLM 的流式推理引擎（生产环境 / x86 + NVIDIA GPU 服务器）。

仅本模块依赖 vllm。若目标机器未安装 vllm，请勿导入本模块——
``server_pt.py`` 会在 vllm 不可用时自动回退到 ``engine_transformers``。

引擎对外接口（与 TransformersStreamEngine 保持一致）：
    - ``stream(prompt_token_ids, stop_event)`` -> 生成器，逐段 yield 新增文本
    - ``shutdown()``
"""
import asyncio
import os
import queue
import threading
import uuid

# 关闭 flashinfer 采样后端：它会在首次推理时用 nvcc 现场 JIT 编译 CUDA 采样
# 算子，从而要求部署机装有匹配 GPU 架构的 CUDA 工具链（否则编译/链接失败）。
# 关掉后改用 vLLM 预编译 kernel，单卡单用户场景性能差异可忽略，却让部署不再
# 依赖运行期编译器。这是后端行为开关（非动态链接路径），用 setdefault 保证外部
# 显式设置仍可覆盖；必须在 import vllm 之前设置，子进程 worker 才能继承。
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


class VLLMStreamEngine:
    """
    在一个后台 asyncio loop 中持有单例 vLLM AsyncLLM。
    socket 处理线程通过同步生成器读取 chunk，实际请求会进入同一个 vLLM 调度器，
    因此天然支持多请求 continuous batching（多用户并发的基础）。
    """

    # vLLM 天然支持多请求并发（continuous batching），服务端无需单请求准入限制。
    SUPPORTS_CONCURRENCY = True

    _END = object()

    def __init__(self, model_dir, tokenizer_dir):
        self.model_dir = model_dir
        self.tokenizer_dir = tokenizer_dir
        self.loop = asyncio.new_event_loop()
        self.engine = None
        self.init_error = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.ready.wait()
        if self.init_error is not None:
            raise self.init_error

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.engine = self._create_engine()
            self.ready.set()
            self.loop.run_forever()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.init_error = exc
            self.ready.set()

    def _create_engine(self):
        # 单卡消费级 GPU（如 16GB）无法容纳 Qwen3 默认 262144 的上下文所需 KV cache，
        # 因此显式限制 max_model_len / gpu_memory_utilization，二者可在 config.yaml 调整。
        from chatlaw.configuration import config
        max_model_len = int(getattr(config, "MAX_MODEL_LEN", 16384) or 16384)
        gpu_util = float(getattr(config, "GPU_MEMORY_UTILIZATION", 0.90) or 0.90)
        # 类比 transformers 的 device_map="auto"：显存装不下权重时把部分权重
        # offload 到 CPU 内存换取可运行性。0 表示不 offload。
        cpu_offload_gb = float(getattr(config, "CPU_OFFLOAD_GB", 0) or 0)

        engine_args = AsyncEngineArgs(
            model=self.model_dir,
            tokenizer=self.tokenizer_dir,
            dtype="auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_util,
            cpu_offload_gb=cpu_offload_gb,
            disable_log_stats=True,
            enable_log_requests=False,
        )
        return AsyncLLM.from_engine_args(engine_args)

    async def _produce(self, request_id, prompt_token_ids, output_queue):
        sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0.7,
            output_kind=RequestOutputKind.DELTA,
            skip_special_tokens=True,
        )
        prompt = {"prompt_token_ids": prompt_token_ids}

        try:
            async for output in self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                for completion in output.outputs:
                    if completion.text:
                        output_queue.put(completion.text)
                if output.finished:
                    break
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            output_queue.put(exc)
        finally:
            output_queue.put(self._END)

    def stream(self, prompt_token_ids, stop_event):
        request_id = f"chatlaw-{uuid.uuid4().hex}"
        output_queue = queue.Queue()
        future = asyncio.run_coroutine_threadsafe(
            self._produce(request_id, prompt_token_ids, output_queue),
            self.loop,
        )
        abort_future = None

        try:
            while True:
                if stop_event.is_set() and abort_future is None:
                    abort_future = asyncio.run_coroutine_threadsafe(
                        self.engine.abort(request_id),
                        self.loop,
                    )

                try:
                    item = output_queue.get(timeout=0.2)
                except queue.Empty:
                    if future.done() and future.exception() is not None:
                        raise future.exception()
                    continue

                if item is self._END:
                    break
                if isinstance(item, BaseException):
                    raise item
                if not stop_event.is_set() and item.strip():
                    yield item
        finally:
            if stop_event.is_set() and abort_future is None and not future.done():
                abort_future = asyncio.run_coroutine_threadsafe(
                    self.engine.abort(request_id),
                    self.loop,
                )
            if abort_future is not None:
                try:
                    abort_future.result(timeout=2)
                except Exception:
                    pass
            # 不再 clear：stop_event 由会话生命周期（SessionHandle）管理，单次使用。

    def shutdown(self):
        if self.engine is not None:
            self.engine.shutdown()
        self.loop.call_soon_threadsafe(self.loop.stop)


__all__ = ["VLLMStreamEngine"]
