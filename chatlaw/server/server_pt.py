"""ChatLaw 服务端入口。

重构后服务端承担全部重活：
    1. 下载并加载全部资源（tokenizer / llm / audio_model / vectorstore）；
    2. 选择推理后端（auto：优先 vllm，不可用则回退 transformers）；
    3. 装配 InferenceService（语音转文字 + 知识库检索 + 分词 + 流式推理）；
    4. 启动心跳服务器 + 数据服务器。

客户端在本架构下不再下载 / 加载任何资源。

注意：所有重资源加载都放在 main() 内，因此单纯 import 本模块不会触发下载或模型加载，
也不会 import vllm —— 这让 service / utils 等模块可以在无 GPU / 无 vllm 的机器上测试。
"""
import os
import threading

from transformers import AutoTokenizer
from funasr_onnx import Paraformer

from chatlaw.dataloader import download_resources
from chatlaw.configuration import config
from chatlaw.server.service import (
    heartbeat_server,
    data_server,
    InferenceService,
    SessionRegistry,
    BusyGate,
)
from chatlaw.server.utils.rag_utils import (
    load_vectorstore,
    load_exact_index,
    build_law_name_candidates,
)
from launcher import get_resources_path


def build_engine(backend, model_path, tokenizer_path):
    """
    根据 backend 选择推理引擎：
        - "vllm"         : 强制 vllm（不可用则抛错）
        - "transformers" : 强制 transformers
        - "auto"（默认） : 优先 vllm，导入 / 初始化失败则回退 transformers

    两种引擎对外接口一致：stream(prompt_token_ids, stop_event) / shutdown()。
    """
    backend = (backend or "auto").lower()

    if backend in ("auto", "vllm"):
        try:
            from chatlaw.server.engine_vllm import VLLMStreamEngine
            print("[Engine] Using vLLM backend.")
            return VLLMStreamEngine(model_path, tokenizer_path)
        except Exception as e:  # pylint: disable=broad-exception-caught
            if backend == "vllm":
                raise
            print(f"[Engine] vLLM unavailable ({e}); falling back to transformers.")

    from chatlaw.server.engine_transformers import TransformersStreamEngine
    print("[Engine] Using transformers backend.")
    return TransformersStreamEngine(model_path, tokenizer_path)


def main():
    resource_path = get_resources_path()

    # === 1. 下载资源（全部在服务端） ===
    print("[Resources] Checking / downloading server resources...")
    download_resources(resource_type="tokenizer")
    download_resources(resource_type="llm")
    download_resources(resource_type="audio_model")
    download_resources(resource_type="vectorstore")

    tokenizer_path = os.path.join(resource_path, "tokenizer").replace("\\", "/")
    model_path = os.path.join(resource_path, "llm").replace("\\", "/")
    audio_model_dir = os.path.join(resource_path, "audio_model").replace("\\", "/")

    # === 2. 加载 tokenizer / ASR / 知识库 ===
    print("[Resources] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    print("[Resources] Loading ASR model (Paraformer)...")
    audio_model = Paraformer(audio_model_dir, batch_size=1, quantize=True, device_id=-1)

    print("[Resources] Loading vectorstore / exact index...")
    vectorstore = load_vectorstore(
        os.path.join(resource_path, "vectorstore", "law_faiss")
    )
    exact_index = load_exact_index()
    law_name_candidates = build_law_name_candidates(exact_index)

    # === 3. 选择并构造推理引擎 ===
    backend = getattr(config, "INFERENCE_BACKEND", "auto")
    engine = build_engine(backend, model_path, tokenizer_path)

    # === 4. 装配 service（多客户端并发） ===
    # 每次咨询用各自会话专属的 stop_event（由 registry 管理），不再共享全局 stop_flag。
    def generate_fn(prompt_token_ids, stop_event):
        yield from engine.stream(prompt_token_ids, stop_event)

    registry = SessionRegistry()
    # 引擎是否支持并发决定准入策略：vLLM 并发放行；transformers 单请求（拒绝并发请求）。
    supports_concurrency = getattr(engine, "SUPPORTS_CONCURRENCY", False)
    gate = BusyGate(concurrent=supports_concurrency)
    print(f"[Server] Backend concurrency: {'enabled (vLLM)' if supports_concurrency else 'single-request (transformers)'}")

    service = InferenceService(
        tokenizer=tokenizer,
        audio_model=audio_model,
        vectorstore=vectorstore,
        exact_index=exact_index,
        law_name_candidates=law_name_candidates,
        generate_fn=generate_fn,
        registry=registry,
        gate=gate,
    )

    # === 5. 起服务 ===
    hb_thread = threading.Thread(
        target=heartbeat_server,
        args=(config.HEARTBEAT_PORT, registry),
        daemon=True,
    )
    hb_thread.start()

    print(
        f"[Server] Ready. heartbeat={config.HEARTBEAT_PORT}, "
        f"data={config.DATA_PORT}"
    )
    try:
        data_server(config.DATA_PORT, service.handle_client)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
