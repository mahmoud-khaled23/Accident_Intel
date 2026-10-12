from pydantic import BaseModel


class LawResult(BaseModel):
    section_id: str
    title: str = ""
    content: str
    score: float = 0.0
    metadata: dict = {}


class LawSearchResponse(BaseModel):
    results: list[LawResult]
    query: str
    total: int
