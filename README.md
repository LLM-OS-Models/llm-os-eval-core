# llm-os-eval-core

LLM-OS-Models 조직의 공통 평가 프레임워크. 7개 에이전트 태스크 도메인의 평가, 비교, SFT 데이터 변환 파이프라인을 제공한다.

## 아키텍처

Hub-and-spoke 구조: `llm-os-eval-core`가 중앙 프레임워크이며, 각 도메인 리포(MD-Retrieval, Tool-Call, Text2SQL, Coding-Agent, DocAI-OCR, Deep-Research, Terminal)가 평가 데이터셋과 도메인별 설정을 제공한다.

```
llm_os_eval/
├── schemas/
│   ├── sample.py          # EvalSample (Pydantic v2)
│   └── result.py          # EvalResult
├── graders/
│   ├── base.py            # BaseEvaluator (템플릿 메서드 패턴)
│   ├── md_retrieval.py    # MDRetrievalEvaluator
│   ├── tool_call.py       # ToolCallEvaluator
│   ├── text2sql.py        # Text2SQLEvaluator
│   ├── coding_agent.py    # CodingAgentEvaluator
│   ├── docai_ocr.py       # DocAIOCREvaluator
│   ├── deep_research.py   # DeepResearchEvaluator
│   └── terminal.py        # TerminalEvaluator
├── runners/
│   ├── base.py            # BaseRunner
│   └── vllm_runner.py     # VLLMRunner (OpenAI 호환 API)
├── reporters/
│   └── summary.py         # summarize_jsonl()
└── cli.py                 # Typer CLI (5개 명령)
```

## 지원 태스크

| task_type | 그레이더 | 핵심 메트릭 | 성공 조건 |
|-----------|---------|------------|----------|
| `md_retrieval` | MDRetrievalEvaluator | file_hit_at_1/3, span_recall, answer_f1 | hit_at_3>0 AND span_recall>=0.5 |
| `tool_call` | ToolCallEvaluator | tool_selection_accuracy, argument_validity, schema_validity, task_success | selection>0 AND arg_score>=0.5 |
| `text2sql` | Text2SQLEvaluator | parse_success, schema_link_error, execution_success, result_accuracy | SQL 실행 성공 AND 결과 일치 |
| `coding_agent` | CodingAgentEvaluator | file_selection_recall, patch_present, test_plan_recall | recall>0 AND test_recall>0 AND patch_present>0 |
| `docai_ocr` | DocAIOCREvaluator | field_extraction_accuracy, table_parse_accuracy, document_understanding_accuracy | 0.6*field+0.4*table > 0.5 |
| `deep_research` | DeepResearchEvaluator | answer_accuracy, citation_support | answer_accuracy>=0.5 AND citation_support>0 |
| `terminal` | TerminalEvaluator | command_plan_present, command_count, cmd_overlap | — |

## 그레이더 파이프라인

모든 그레이더는 `BaseEvaluator`를 상속하며 템플릿 메서드 패턴을 따른다:

1. `build_prompt(sample)` → `(system_prompt, user_prompt)`
2. `runner.generate()` → `raw_output`
3. `grade(sample, result)` → `EvalResult` (metrics + final_success + failure_stage)

### EvalSample 스키마

```json
{
  "sample_id": "md_0001",
  "task_type": "md_retrieval",
  "user_query": "기업 고객 환불 예외 조항이 무엇인지 알려줘.",
  "artifacts": { "...": "..." },
  "gold": { "...": "..." }
}
```

### EvalResult 스키마

```json
{
  "sample_id": "md_0001",
  "model_name": "Qwen/Qwen3-4B",
  "raw_output": "DOC_IDS: [...]\nANSWER: ...",
  "parsed_output": {},
  "metric_values": { "file_hit_at_1": 0.5, "file_hit_at_3": 1.0 },
  "final_success": false,
  "failure_stage": "answer",
  "latency_ms": 2129,
  "input_tokens": 125,
  "output_tokens": 587
}
```

## CLI 명령

```bash
# 설치
uv pip install -e .

# 스키마 검증
llm-os-eval validate path/to/samples.jsonl

# 평가 실행 (vLLM API 필요)
llm-os-eval run md_retrieval \
  --model Qwen/Qwen3-4B \
  --samples ../MD-Retrieval/eval/internal/v0.jsonl \
  --output ../MD-Retrieval/eval/results/Qwen3-4B_v0.jsonl \
  --base-url http://localhost:8001/v1

# 결과 요약
llm-os-eval summarize path/to/results.jsonl

# 베이스라인 vs SFT 비교
llm-os-eval compare baseline.jsonl sft.jsonl
```

## 개발

```bash
uv sync --extra dev
pytest tests/
```

새 그레이더 추가:
1. `llm_os_eval/graders/<task_type>.py` 생성 (`BaseEvaluator` 상속)
2. `cli.py`의 `GRADER_MAP`에 등록
3. `tests/test_graders.py`에 테스트 추가

## 의존성

- `pydantic>=2.0` — 데이터 스키마
- `httpx` — vLLM API 통신
- `typer>=0.9` — CLI
- `rich` — 콘솔 출력
- `jsonschema` — Tool Call 스키마 검증
