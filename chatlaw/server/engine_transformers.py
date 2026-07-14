"""基于 HuggingFace transformers 的流式推理引擎。

用途：当目标机器未安装 vllm（例如 Windows 开发机）时，作为等价替代，
让整条服务链路（STT -> RAG -> 分词 -> 推理流式输出）可以真实跑通并测试。
生产环境仍应使用 ``engine_vllm.VLLMStreamEngine``。

对外接口与 VLLMStreamEngine 保持一致：
    - ``stream(prompt_token_ids, stop_event)`` -> 生成器，逐段 yield 新增文本
    - ``shutdown()``

注意：transformers 的 ``model.generate`` 不是并发安全的，这里用一把锁把生成串行化，
因此该引擎实际上一次只服务一个请求。多用户真正的并发交给 vLLM（后续步骤）。
"""
import threading

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)


class _StopOnEvent(StoppingCriteria):
    """当外部 stop_event 被置位时，让 generate 在下一个 token 处停止。"""

    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):  # noqa: D401
        return self.stop_event.is_set()


class TransformersStreamEngine:
    def __init__(self, model_dir, tokenizer_dir, max_new_tokens=4096):
        self.model_dir = model_dir
        self.tokenizer_dir = tokenizer_dir
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            local_files_only=True,
        )
        self.model.eval()

        # 单 GPU / CPU 场景下，取模型参数所在设备用于放置输入
        self.device = next(self.model.parameters()).device

        # transformers generate 非并发安全，串行化
        self._lock = threading.Lock()

    def stream(self, prompt_token_ids, stop_event):
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        input_ids = torch.tensor(
            [list(prompt_token_ids)], dtype=torch.long, device=self.device
        )
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([_StopOnEvent(stop_event)]),
        )

        with self._lock:
            error_box = {}

            def _run():
                try:
                    with torch.no_grad():
                        self.model.generate(**gen_kwargs)
                except BaseException as exc:  # pylint: disable=broad-exception-caught
                    error_box["error"] = exc
                finally:
                    # 保证消费端不会因异常/中断而永久阻塞在 streamer 上
                    try:
                        streamer.end()
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass

            gen_thread = threading.Thread(target=_run, daemon=True)
            gen_thread.start()

            try:
                for new_text in streamer:
                    if stop_event.is_set():
                        break
                    if new_text:
                        yield new_text
            finally:
                gen_thread.join(timeout=30)
                stop_event.clear()

            if error_box.get("error") is not None:
                raise error_box["error"]

    def shutdown(self):
        # transformers 无需显式关闭；释放引用即可
        self.model = None


__all__ = ["TransformersStreamEngine"]
