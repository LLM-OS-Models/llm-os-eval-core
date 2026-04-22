from __future__ import annotations
import sqlite3
from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult

class Text2SQLEvaluator(BaseEvaluator):
    task_type = "text2sql"

    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        system_prompt = "너는 text-to-sql 평가용 모델이다. 반드시 SQL만 출력하라."
        schema_docs = sample.artifacts.get("schema_docs", [])
        user_prompt = f"""질문:
{sample.user_query}

DB:
{sample.artifacts.get('db_path')}

문서:
{schema_docs}

출력:
SQL 한 개만 출력
"""
        return system_prompt, user_prompt

    def grade(self, sample: EvalSample, result: EvalResult) -> EvalResult:
        sql = (result.raw_output or "").strip().strip("```sql").strip("```")
        db_path = sample.artifacts.get("db_path")
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            conn.close()
            result.metric_values["parse_success"] = 1.0
            result.metric_values["execution_success"] = 1.0
            result.final_success = True if rows is not None else False
        except Exception:
            result.metric_values["parse_success"] = 0.0
            result.metric_values["execution_success"] = 0.0
            result.final_success = False
            result.failure_stage = "sql_execution"
        return result
