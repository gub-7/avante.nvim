from pydantic import BaseModel


class RagEvalCase(BaseModel):
    id: str
    query: str
    mode: str
    base_uri: str
    current_file: str | None = None
    latest_error: str | None = None
    expected_files: list[str]
    expected_symbols: list[str] = []
    must_not_retrieve: list[str] = []


class EvalRunResult(BaseModel):
    case_id: str
    recall_at_k: float
    precision_at_k: float
    mrr: float
    expected_symbol_hit_rate: float
    irrelevant_context_tokens: int
    inserted_token_count: int
    freshness_error_rate: float
    dedupe_savings: int


class EvalReport(BaseModel):
    results: list[EvalRunResult]
    aggregate: dict[str, float]
    trace_ids: list[str] = []

