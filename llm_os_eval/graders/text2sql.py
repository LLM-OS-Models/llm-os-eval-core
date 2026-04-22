from __future__ import annotations
import hashlib
import re
import sqlite3

from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult


def _extract_sql(text: str) -> str:
    sql = text.strip()
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL)
    if fence_match:
        sql = fence_match.group(1).strip()
    for prefix in ("SELECT", "select", "Select", "WITH", "with", "INSERT", "UPDATE", "DELETE"):
        idx = sql.upper().find(prefix)
        if idx >= 0:
            sql = sql[idx:]
            break
    sql = sql.rstrip(";").strip()
    return sql


def _row_hash(rows: list[tuple]) -> str:
    serialized = "|".join(str(sorted(row)) for row in sorted(rows))
    return hashlib.md5(serialized.encode()).hexdigest()


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
        text = result.raw_output or ""
        sql = _extract_sql(text)
        db_path = sample.artifacts.get("db_path")
        gold_result_hash = sample.gold.get("result_hash", None)
        gold_sql_patterns = sample.gold.get("acceptable_sql_patterns", [])

        if not sql:
            result.metric_values["parse_success"] = 0.0
            result.metric_values["execution_success"] = 0.0
            result.metric_values["result_accuracy"] = 0.0
            result.final_success = False
            result.failure_stage = "sql_empty"
            return result

        result.metric_values["parse_success"] = 1.0

        for pattern in gold_sql_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                result.metric_values["schema_link_error"] = 0.0
                break
        else:
            if gold_sql_patterns:
                result.metric_values["schema_link_error"] = 1.0

        if not db_path:
            result.metric_values["execution_success"] = 1.0
            result.metric_values["result_accuracy"] = 1.0
            result.final_success = True
            return result

        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
            result.metric_values["execution_success"] = 1.0

            if gold_result_hash:
                pred_hash = _row_hash(rows) if rows else ""
                result.metric_values["result_accuracy"] = 1.0 if pred_hash == gold_result_hash else 0.0
            else:
                result.metric_values["result_accuracy"] = 1.0 if rows is not None and len(rows) > 0 else 0.0

            result.final_success = result.metric_values["result_accuracy"] > 0
        except Exception as e:
            result.metric_values["execution_success"] = 0.0
            result.metric_values["result_accuracy"] = 0.0
            result.final_success = False
            result.failure_stage = "sql_execution"
            result.metadata["error"] = str(e)
        return result
