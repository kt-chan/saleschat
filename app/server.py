from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app.online import suggest_script_for_conversation, get_retriever

app = FastAPI(title="催收话术推荐")


@app.on_event("startup")
def _warm():
    get_retriever()


class SuggestReq(BaseModel):
    conversation: str
    stage_hint: Optional[str] = None


@app.post("/suggest")
def suggest(req: SuggestReq):
    return suggest_script_for_conversation(
        req.conversation, stage_hint=req.stage_hint,
    )


@app.get("/health")
def health():
    return {"status": "ok"}