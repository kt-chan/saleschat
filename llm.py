import re, time, yaml
from typing import Dict, Optional

from openai import OpenAI

from settings import get_settings
from logger import get_logger
from utils import safe_json_loads

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
        self.client = OpenAI(
            api_key=self.cfg.llm_api_key,
            base_url=self.cfg.llm_base_url,
            timeout=self.cfg.llm_timeout,
        )
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
        response_format: Optional[str] = "json_object",
    ) -> Dict:
        temperature = (
            self.cfg.llm_temperature_labeling if temperature is None else temperature
        )
        last_exc: Optional[Exception] = None
        for attempt in range(self.cfg.llm_retry_max):
            try:
                kwargs = dict(
                    model=self.cfg.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                if response_format:
                    kwargs["response_format"] = {"type": response_format}
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                log.debug(f"prompt[:16000]: {prompt[:16000]}")
                log.debug(f"response[:2000]: {content[:2000]}")
                return safe_json_loads(content)
            except Exception as e:
                last_exc = e
                if attempt < self.cfg.llm_retry_max - 1:
                    time.sleep(self.cfg.llm_retry_backoff ** attempt)
        raise last_exc


_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client