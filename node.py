"""
abliteration_node — abliteration.ai(无审查推理 API)的 ComfyUI 节点。

为什么自己写而不是复用 openrouter_node:
  1. openrouter_node 的 base URL 硬编码到 openrouter.ai,换不了服务商。
  2. 更关键 —— 它在上游出错/被安全过滤截断时**静默返回残缺文本**。实测 Gemini 会
     间歇性 finish_reason=error / native=SAFETY,节点照样把半句话往下传,视频模型
     拿到残缺(甚至空)提示词就自由发挥:主体跑偏、糊上一堆图形。这种失败必须**炸**,
     不能悄悄污染下游。本节点在以下情况直接 raise,让运行停在这里:
       - HTTP 非 2xx
       - finish_reason 不是 stop(length=被 max_tokens 截断,其它=被过滤/出错)
       - content 为空
  3. 只用标准库 urllib 发请求(不引入 requests/tiktoken),Modal 容器里少一层依赖风险。

API:OpenAI 兼容,POST https://api.abliteration.ai/v1/chat/completions,Bearer 鉴权。
模型:abliterated-model(多模态,256K)/ abliterated-model-large(纯文本,1M —— 传图会 400)。
推理:两个模型默认都会思考;思考轨迹走 message.reasoning_content,不会混进 content。
      reasoning_effort=none 可完全关掉(省 token / 更快)。
"""
import base64
import hashlib
import io
import json
import time
import urllib.error
import urllib.request

try:
    from server import PromptServer   # 只有 ComfyUI 进程里才有
except Exception:                     # 纯 Python 环境(测试/脚本)照样能 import 本模块
    PromptServer = None

API_URL = "https://api.abliteration.ai/v1/chat/completions"
MODELS = ["abliterated-model", "abliterated-model-large"]
EFFORT = ["none", "minimal", "low", "medium", "high", "xhigh"]


class _StreamUnsupported(Exception):
    """上游明确拒绝 stream 参数 —— 触发一次性请求回退,不是错误。"""


def _emit_text(unique_id, text: str, done: bool) -> None:
    """把部分/最终文本广播到 ComfyUI ws(自定义事件 abliteration.text)。

    本地跑:前端 modal_bridge.js 监听同名事件,实时刷新 Display Any 节点;
    云端跑:worker 的 ws 客户端(_comfy_ws.run_workflow)收到后写进 job_state.node_text,
           由 /modal_bridge/poll 原样透传回本地前端。
    发送失败一律静默 —— 直播是锦上添花,绝不能影响任务本体。"""
    if PromptServer is None or unique_id is None:
        return
    try:
        PromptServer.instance.send_sync(
            "abliteration.text", {"node": str(unique_id), "text": text, "done": done})
    except Exception:
        pass


def _parse_sse_data(line: str):
    """SSE 单行 → dict | "[DONE]" | None(空行 / : 注释 / 非 data 行 / 坏 JSON)。"""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return "[DONE]"
    try:
        return json.loads(payload)
    except Exception:
        return None


def _make_request(payload: dict, key: str) -> "urllib.request.Request":
    return urllib.request.Request(
        API_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST")


def _http_error_detail(e) -> str:
    try:
        return e.read().decode("utf-8", "replace")[:600]
    except Exception:
        return ""


def _tensor_to_data_uri(image) -> str:
    """ComfyUI IMAGE tensor (B,H,W,C) float 0-1 → data:image/jpeg;base64,...

    只取第一张:参考图语义上就是一张(<Picture 1>)。JPEG 而不是 PNG —— 上游限制
    单图 12MB,4096 宽的 PNG 很容易顶到上限。"""
    import numpy as np
    from PIL import Image

    arr = image[0].cpu().numpy() if hasattr(image[0], "cpu") else np.asarray(image[0])
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype("uint8")
    pil = Image.fromarray(arr)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class AbliterationChat:
    """abliteration.ai chat completion。返回模型正文(STRING)。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "user_message_box": ("STRING", {"multiline": True, "default": ""}),
                "model": (MODELS, {"default": "abliterated-model"}),
                # ⚠ 默认 none:max_tokens 是**含推理 token 的总预算**。这两个模型默认都会思考,
                # 推理会先吃预算,吃光就出现 finish_reason=length 且 content 一个字都没有
                # (实测 effort=low + max_tokens=4096 就这样)。改写提示词是格式活,不需要思考。
                "reasoning_effort": (EFFORT, {"default": "none"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 65536}),
                "timeout_sec": ("INT", {"default": 180, "min": 10, "max": 900}),
            },
            "optional": {
                # forceInput:这两个是给上游连线用的,不在节点上显示输入框
                "user_message_input": ("STRING", {"forceInput": True}),
                "image_1": ("IMAGE",),
            },
            # unique_id:节点自报家门,_emit_text 靠它告诉前端"这段文本属于哪个节点"
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Output",)
    FUNCTION = "generate"
    CATEGORY = "LLM"

    def generate(self, api_key, system_prompt, user_message_box, model, reasoning_effort,
                 temperature, max_tokens, timeout_sec,
                 user_message_input=None, image_1=None, unique_id=None, **kwargs):
        key = (api_key or "").strip()
        if not key:
            raise RuntimeError(
                "AbliterationChat: api_key 为空。把 abliteration.ai 的 key 填进 api_key 输入"
                "(或连一个 StringConstant 进来)。")

        # 连线的 user_message_input 优先于框里手打的内容(和 openrouter_node 语义一致)
        user_text = (user_message_input if user_message_input not in (None, "") else user_message_box) or ""
        if not user_text.strip():
            raise RuntimeError("AbliterationChat: user message 为空,拒绝发空请求。")

        if image_1 is not None and model == "abliterated-model-large":
            raise RuntimeError(
                "AbliterationChat: abliterated-model-large 是纯文本模型,传图会被上游拒绝(400)。"
                "要用参考图请选 abliterated-model。")

        if image_1 is None:
            content = user_text
        else:
            content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": _tensor_to_data_uri(image_1)}},
            ]

        payload = {
            "model": model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "reasoning_effort": reasoning_effort,
            "include_reasoning": False,   # 思考轨迹不要回传,省带宽(不省 token)
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": content},
            ],
        }

        try:
            text, finish, usage, native = self._generate_stream(
                payload, key, int(timeout_sec), unique_id)
        except _StreamUnsupported as e:
            print(f"[abliteration] 上游拒绝流式({e}),回退一次性请求 —— 直播只剩最终文本。")
            text, finish, usage, native = self._generate_blocking(payload, key, int(timeout_sec))

        # ⚠ 这几个检查是本节点存在的理由:上游一旦截断/被拦,宁可炸掉整次运行,
        #   也不能把残缺提示词喂给视频模型(那会烧掉一次 GPU 还产出废片)。
        reasoning_tok = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        if finish == "length":
            # 区分两种完全不同的 length:正文被截断 vs 推理把预算吃光(正文 0 字)。
            # 前者调大 max_tokens 有用;后者必须关推理,调大只是烧更多钱。
            if not text.strip():
                raise RuntimeError(
                    f"AbliterationChat: max_tokens={max_tokens} 全被**推理**吃掉了,正文 0 字符"
                    f"(completion_tokens={usage.get('completion_tokens')}, "
                    f"reasoning_tokens={reasoning_tok})。\n"
                    f"max_tokens 是含推理 token 的总预算,而 include_reasoning=false 只是不回传轨迹、"
                    f"并不省 token。\n"
                    f"解法:把 reasoning_effort 设为 none(当前 {reasoning_effort!r})—— 改写提示词"
                    f"是格式活,不需要思考;或者把 max_tokens 调到 16000 以上给正文留空间。")
            raise RuntimeError(
                f"AbliterationChat: 正文被 max_tokens={max_tokens} 截断(finish_reason=length),"
                f"已生成 {len(text)} 字符(reasoning_tokens={reasoning_tok})。把 max_tokens 调大再跑。")
        if finish not in ("stop", None):
            raise RuntimeError(
                f"AbliterationChat: 生成未正常结束,finish_reason={finish!r} "
                f"(native={native!r})。已生成 {len(text)} 字符,不往下传。")
        if not text.strip():
            raise RuntimeError("AbliterationChat: 模型返回空正文,不往下传。")

        # 最终文本(两条路径共用这一次 done 广播):本地前端直接刷新 Display Any,
        # 云端由 worker ws → job_state.node_text → poll 回传。
        _emit_text(unique_id, text, True)

        # 整段照旧打到 stdout:上面的 ws 直播只是 UI 显示,worker 日志才是事后可查的存档
        # (modal app logs comfyui-bridge),排查"这次到底喂了什么提示词"仍然靠它。
        print(f"[abliteration] {model} effort={reasoning_effort} "
              f"in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')} "
              f"reasoning={reasoning_tok} chars={len(text)} finish={finish}")
        print("[abliteration] ===== GENERATED PROMPT BEGIN =====")
        for line in text.splitlines():
            print(f"[abliteration] {line}")
        print("[abliteration] ===== GENERATED PROMPT END =====")
        return (text,)

    def _generate_stream(self, payload, key, timeout_sec, unique_id):
        """SSE 流式请求。边收边广播部分文本,返回 (text, finish, usage, native)。"""
        body = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        try:
            resp = urllib.request.urlopen(_make_request(body, key), timeout=timeout_sec)
        except urllib.error.HTTPError as e:
            detail = _http_error_detail(e)
            if e.code == 400 and "stream" in detail.lower():
                raise _StreamUnsupported(detail[:200].replace("\n", " ")) from None
            raise RuntimeError(
                f"AbliterationChat: HTTP {e.code} from abliteration.ai — {detail}") from None
        except Exception as e:
            raise RuntimeError(f"AbliterationChat: 请求失败 — {type(e).__name__}: {e}") from None

        text_parts, finish, native, usage = [], None, None, {}
        saw_done = False
        last_emit_len, last_emit_t = 0, 0.0
        try:
            with resp:
                for raw in resp:
                    evt = _parse_sse_data(raw.decode("utf-8", "replace"))
                    if evt is None:
                        continue
                    if evt == "[DONE]":
                        saw_done = True
                        break
                    for ch in (evt.get("choices") or []):
                        delta = (ch.get("delta") or {}).get("content")
                        if isinstance(delta, str) and delta:
                            text_parts.append(delta)
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
                            native = ch.get("native_finish_reason")
                    if evt.get("usage"):
                        usage = evt["usage"]
                    # 限频直播:新增 ≥120 字符,或距上次 ≥0.35s,才广播一次
                    cur = "".join(text_parts)
                    now = time.time()
                    if len(cur) - last_emit_len >= 120 or (cur and now - last_emit_t >= 0.35):
                        last_emit_len, last_emit_t = len(cur), now
                        _emit_text(unique_id, cur, False)
        except Exception as e:
            raise RuntimeError(f"AbliterationChat: 流式读取中断 — {type(e).__name__}: {e}") from None

        text = "".join(text_parts)
        if finish is None and not saw_done:
            # 连接断在半路:没有 finish_reason 也没有 [DONE],和"正常收完"无法区分,只能炸。
            raise RuntimeError(
                f"AbliterationChat: SSE 流中断,未收到 finish_reason,不往下传。已收 {len(text)} 字符。")
        return text, finish, usage, native

    def _generate_blocking(self, payload, key, timeout_sec):
        """一次性请求(不流式)—— 只在上游拒绝 stream 时走这条回退路径。"""
        try:
            with urllib.request.urlopen(_make_request(payload, key), timeout=timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"AbliterationChat: HTTP {e.code} from abliteration.ai — {_http_error_detail(e)}") from None
        except Exception as e:
            raise RuntimeError(f"AbliterationChat: 请求失败 — {type(e).__name__}: {e}") from None

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"AbliterationChat: 响应里没有 choices — {json.dumps(body)[:400]}")
        ch = choices[0]
        return ((ch.get("message") or {}).get("content") or "",
                ch.get("finish_reason"), body.get("usage") or {}, ch.get("native_finish_reason"))

    @classmethod
    def IS_CHANGED(cls, api_key, system_prompt, user_message_box, model, reasoning_effort,
                   temperature, max_tokens, timeout_sec,
                   user_message_input=None, image_1=None, unique_id=None, **kwargs):
        """任一输入变了就重新执行。图片按内容摘要参与判定 —— 否则换了参考图
        ComfyUI 会命中缓存、继续用上一次的提示词(这类"改了没反应"极难排查)。"""
        h = hashlib.sha256()
        for part in (system_prompt, user_message_box, model, reasoning_effort,
                     str(temperature), str(max_tokens), user_message_input or ""):
            h.update(str(part).encode("utf-8", "replace"))
            h.update(b"\x00")
        if image_1 is not None:
            try:
                arr = image_1[0]
                buf = arr.cpu().numpy().tobytes() if hasattr(arr, "cpu") else bytes(memoryview(arr))
                h.update(hashlib.sha256(buf).digest())
            except Exception:
                h.update(b"image-unhashable")
        return h.hexdigest()


NODE_CLASS_MAPPINGS = {"AbliterationChat": AbliterationChat}
NODE_DISPLAY_NAME_MAPPINGS = {"AbliterationChat": "Abliteration.ai LLM (text + image)"}
