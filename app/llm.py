import requests
from requests.exceptions import HTTPError, Timeout
import re, time, yaml
from typing import Dict, Optional
from app.settings import get_settings
from app.logger import get_logger
from app.utils import safe_json_loads

log = get_logger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_ESCAPE_OPEN = "\x00EO\x00"
_ESCAPE_CLOSE = "\x00EC\x00"


def render_prompt(template: str, **kwargs) -> str:
    """Substitute {key} placeholders; {{ }} escapes to literal braces."""
    escaped = template.replace("{{", _ESCAPE_OPEN).replace("}}", _ESCAPE_CLOSE)

    def _repl(m):
        key = m.group(1)
        return str(kwargs[key]) if key in kwargs else m.group(0)

    result = _PLACEHOLDER_RE.sub(_repl, escaped)
    return result.replace(_ESCAPE_OPEN, "{").replace(_ESCAPE_CLOSE, "}")


class LLMClient:
    def __init__(self, cfg=None):
        self.cfg = cfg or get_settings()

        # 1. 拼接 URL
        base = self.cfg.llm_base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            self.url = base
        else:
            self.url = f"{base}/chat/completions"

        # 2. 构造 Headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.llm_api_key}",
        }
        # 核心魔法：只有存在 Host 配置时才注入（应对 CMB）
        if getattr(self.cfg, "llm_host", None):
            self.headers["Host"] = self.cfg.llm_host

        # 3. 加载 Prompt
        with open(self.cfg.prompt_file, "r", encoding="utf-8") as f:
            self.prompts: Dict = yaml.safe_load(f) or {}

    def render(self, template_key: str, **kwargs) -> str:
        template = self.prompts.get(template_key)
        if not template:
            raise KeyError(f"Prompt template not found: {template_key}")
        return render_prompt(template, **kwargs)

    def call(
        self,
        prompt: str,
        temperature: Optional[float] = None,
    ) -> Dict:
        temp_val = (
            self.cfg.llm_temperature_labeling if temperature is None else temperature
        )

        # 默认尝试使用 JSON Mode
        use_json_format = True

        last_exc: Optional[Exception] = None

        for attempt in range(self.cfg.llm_retry_max):
            try:
                payload = {
                    "model": self.cfg.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temp_val,
                }
                if use_json_format:
                    payload["response_format"] = {"type": "json_object"}

                with requests.Session() as session:
                    resp = session.post(
                        self.url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.cfg.llm_timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content") or ""
                )

                log.debug(f"prompt[:16000]: {prompt[:16000]}")
                log.debug(f"response[:2000]: {content[:2000]}")

                return safe_json_loads(content)

            except HTTPError as e:
                last_exc = e
                # 核心降级逻辑：如果是 400 错误，且当前是开启 JSON Mode 的状态
                if e.response.status_code == 400 and use_json_format:
                    log.warning(
                        f"模型不支持 response_format 参数，自动降级并重试 (报错: {e.response.text})"
                    )
                    use_json_format = False
                    continue  # 立即重试，不进行 sleep 等待

                # 其他 HTTP 错误，走标准退避重试
                log.warning(f"LLM API HTTP 错误 (第 {attempt + 1} 次尝试): {e}")
                if attempt < self.cfg.llm_retry_max - 1:
                    time.sleep(self.cfg.llm_retry_backoff**attempt)

            except Exception as e:
                last_exc = e
                log.warning(f"LLM API 调用异常 (第 {attempt + 1} 次尝试): {e}")
                if attempt < self.cfg.llm_retry_max - 1:
                    time.sleep(self.cfg.llm_retry_backoff**attempt)

        raise last_exc


_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
