# llm-os-eval-core

LLM-OS-Models 조직의 공통 실험 하네스입니다. 7개 에이전트 태스크 도메인의 평가·학습 파이프라인을 제공합니다.

## 지원 태스크

| task_type | 프로젝트 | 그레이더 | 핵심 메트릭 |
|-----------|---------|---------|------------|
| `md_retrieval` | MD-Retrieval | MDRetrievalEvaluator | file_hit_at_1, file_hit_at_3, span_recall, answer_f1 |
| `tool_call` | Tool-Call | ToolCallEvaluator | tool_selection_accuracy, argument_validity, schema_validity, task_success |
| `terminal` | Terminal | TerminalEvaluator | command_plan_present, command_count, cmd_overlap |
| `text2sql` | Text2SQL | Text2SQLEvaluator | parse_success, schema_link_error, execution_success, result_accuracy |
| `coding_agent` | Coding-Agent | CodingAgentEvaluator | file_selection_recall, patch_present, test_plan_recall |
| `docai_ocr` | DocAI-OCR | DocAIOCREvaluator | field_extraction_accuracy, table_parse_accuracy, document_understanding_accuracy |
| `deep_research` | Deep-Research | DeepResearchEvaluator | answer_accuracy, citation_support |

## CLI 명령

```bash
# 헬스체크
llm-os-eval hello

# 샘플 파일 스키마 검증
llm-os-eval validate <samples.jsonl>

# 평가 실행 (vLLM API)
llm-os-eval run --task-type <type> --model <model> --samples <jsonl> --output <path>

# 결과 요약
llm-os-eval summarize --results-dir <dir>

# 베이스라인 vs SFT 비교
llm-os-eval compare <baseline.jsonl> <sft.jsonl>
```

## 아키텍처

```
llm_os_eval/
├── schemas/          # EvalSample, EvalResult Pydantic 모델
├── graders/          # 7개 태스크별 그레이더 (BaseEvaluator 상속)
├── runners/          # VLLMRunner, TransformersRunner
├── reporters/        # 결과 요약 리포터
├── cli.py            # CLI 진입점 (5개 명령)
├── eval_runner.py    # transformers 기반 단일 GPU 평가 스크립트
├── shared_summarize.py  # 결과 집계 스크립트
├── prepare_sft.py    # 평가 샘플 → SFT 학습 데이터 변환
└── sft_train.py      # LoRA SFT 학습 스크립트
```

## 그레이더 파이프라인

모든 그레이더는 `BaseEvaluator`를 상속하며 템플릿 메서드 패턴을 따릅니다:

1. `build_prompt(sample)` → (system_prompt, user_prompt)
2. `runner.generate()` → raw_output
3. `grade(sample, result)` →EvalResult (메트릭 계산)
