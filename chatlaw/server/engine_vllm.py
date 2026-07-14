"""基于 vLLM AsyncLLM 的流式推理引擎（生产环境 / x86 + NVIDIA GPU 服务器）。

仅本模块依赖 vllm。若目标机器未安装 vllm，请勿导入本模块——
``server_pt.py`` 会在 vllm 不可用时自动回退到 ``engine_transformers``。

引擎对外接口（与 TransformersStreamEngine 保持一致）：
    - ``stream(prompt_token_ids, stop_event)`` -> 生成器，逐段 yield 新增文本
    - ``shutdown()``
"""
import asyncio
import queue
import threading
import uuid

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
        engine_args = AsyncEngineArgs(
            model=self.model_dir,
            tokenizer=self.tokenizer_dir,
            dtype="auto",
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
            stop_event.clear()

    def shutdown(self):
        if self.engine is not None:
            self.engine.shutdown()
        self.loop.call_soon_threadsafe(self.loop.stop)


__all__ = ["VLLMStreamEngine"]
