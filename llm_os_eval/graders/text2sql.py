from __future__ import annotations
import hashlib
import re
import sqlite3

from llm_os_eval.graders.base import BaseEvaluator
from llm_os_eval.schemas.sample import EvalSample
from llm_os_eval.schemas.result import EvalResult


def _extract_sql(text: str) -> str:
    sql = text.strip()
    # Strip <think...>...</think...> closed blocks
    sql = re.sub(r"<think[^>]*>.*?</think\s*>", "", sql, flags=re.DOTALL)
    # Strip unclosed thinking prefix: find first line starting with fence or SQL keyword
    lines = sql.split("\n")
    content_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            content_start = i
            break
        for kw in ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE"):
            if stripped.upper().startswith(kw):
                content_start = i
                break
        if content_start is not None:
            break
    if content_start is not None and content_start > 0:
        sql = "\n".join(lines[content_start:])
    # Try fenced code block first (most reliable)
    # Note: longer alternatives first to avoid "sql" matching prefix of "sqlite"
    fence_match = re.search(r"```(?:sqlite|SQL|sql)?\s*\n?(.*?)```", sql, re.DOTALL)
    if fence_match:
        sql = fence_match.group(1).strip()
    else:
        # Only search for SQL keywords after the last newline that starts a SQL-looking line
        # to avoid matching "select" in reasoning text
        for line in sql.split("\n"):
            stripped = line.strip()
            for kw in ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE"):
                if stripped.upper().startswith(kw):
                    idx = sql.find(line)
                    sql = sql[idx:]
                    break
            else:
                continue
            break
    sql = sql.rstrip(";").strip()
    # Only take the first statement if multiple are present
    if ";" in sql:
        sql = sql.split(";")[0].strip()
    return sql


def _row_hash(rows: list[tuple]) -> str:
    serialized = "|".join(
        str(sorted([str(c) for c in row]))
        for row in sorted(rows, key=lambda x: str(x))
    )
    return hashlib.md5(serialized.encode()).hexdigest()[:6]


def _normalize_row(row: tuple) -> frozenset:
    return frozenset(str(c).strip().lower() for c in row)


def _rows_match(pred_row: frozenset, gold_row: frozenset) -> bool:
    if pred_row == gold_row:
        return True
    if gold_row <= pred_row:
        return True
    overlap = len(pred_row & gold_row)
    if overlap == 0:
        return False
    return overlap / max(len(pred_row), len(gold_row)) >= 0.8


def _compute_result_f1(pred_rows: list[tuple], gold_rows: list[tuple]) -> float:
    if not pred_rows and not gold_rows:
        return 1.0
    if not pred_rows or not gold_rows:
        return 0.0
    pred_set = {_normalize_row(r) for r in pred_rows}
    gold_set = {_normalize_row(r) for r in gold_rows}
    recall_hits = sum(1 for g in gold_set if any(_rows_match(p, g) for p in pred_set))
    precision_hits = sum(1 for p in pred_set if any(_rows_match(p, g) for g in gold_set))
    precision = precision_hits / len(pred_set) if pred_set else 0.0
    recall = recall_hits / len(gold_set) if gold_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class Text2SQLEvaluator(BaseEvaluator):
    task_type = "text2sql"

    def build_prompt(self, sample: EvalSample) -> tuple[str, str]:
        system_prompt = "너는 text-to-sql 평가용 모델이다. 사고 과정이나 설명 없이 SQL 쿼리만 즉시 출력하라. <think:> 태그를 절대 사용하지 마라. 반드시 SQLite 호환 문법만 사용하라. DATE_SUB, CURDATE(), NOW() 등 MySQL 함수는 사용하지 마라. 대신 date(), datetime(), date('now') 등 SQLite 함수를 사용하라. 중요: SQLite에서 정수/정수는 정수 나눗셈이다. 비율 계산 시 반드시 100.0을 곱하거나 CAST(x AS REAL)을 사용하라. 예: SUM(col)*100.0/COUNT(*) (O), SUM(col)/COUNT(*) (X). date() 함수에서 N은 반드시 구체적인 숫자로 치환하라. 예: date('now','-30 days') (O), date('now','-N days') (X)."
        db_path = sample.artifacts.get("db_path")
        schema_docs = sample.artifacts.get("schema_docs", [])

        # Extract actual schema from DB if path exists
        schema_info = ""
        if db_path:
            try:
                with sqlite3.connect(db_path) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cur.fetchall()]
                    schema_lines = []
                    for tbl in tables:
                        cur.execute(f"PRAGMA table_info({tbl})")
                        cols = [f"{row[1]} {row[2]}" for row in cur.fetchall()]
                        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                        cnt = cur.fetchone()[0]
                        schema_lines.append(f"TABLE {tbl} ({', '.join(cols)}) -- {cnt} rows")
                        cur.execute(f"SELECT * FROM {tbl} LIMIT 2")
                        sample_rows = cur.fetchall()
                        col_names = [d[1] for d in cur.description] if cur.description else []
                        for row in sample_rows:
                            schema_lines.append(f"  예시: {dict(zip(col_names, row))}")
                    schema_info = "\n".join(schema_lines)
            except Exception:
                pass

        user_prompt = f"""질문:
{sample.user_query}

DB 스키마:
{schema_info}

참고 문서:
{schema_docs}

출력:
SQL 한 개만 즉시 출력하라. 설명이나 주석 없이 SQL만 출력하라.
반드시 위 스키마의 테이블명과 컬럼명을 사용하라.
SELECT * 대신 질문에 필요한 컬럼만 선택하라. 날짜 비교에는 date('now', '-N days') 형식을 사용하라.
중요: WHERE 절의 값은 예시 데이터의 실제 값을 그대로 사용하라. 번역하지 마라. (예: 'premium'을 '프리미엄'으로 변환 금지, 대소문자도 예시와 동일하게)
"""
        return system_prompt, user_prompt

    def grade(self, sample: EvalSample, result: EvalResult) -> EvalResult:
        text = result.raw_output or ""
        sql = _extract_sql(text)
        db_path = sample.artifacts.get("db_path")
        gold_result_hash = sample.gold.get("result_hash", None)
        gold_result_rows = sample.gold.get("gold_result", None)
        gold_sql = sample.gold.get("gold_sql", None)
        gold_sql_patterns = sample.gold.get("acceptable_sql_patterns", [])

        if not sql:
            result.metric_values["parse_success"] = 0.0
            result.metric_values["execution_success"] = 0.0
            result.metric_values["result_accuracy"] = 0.0
            result.metric_values["result_f1"] = 0.0
            result.final_success = False
            result.failure_stage = "sql_empty"
            return result

        result.metric_values["parse_success"] = 1.0
        result.metric_values["schema_link_error"] = 0.0

        if gold_sql_patterns:
            matched = False
            for pattern in gold_sql_patterns:
                if re.search(pattern, sql, re.IGNORECASE):
                    matched = True
                    break
            if not matched:
                result.metric_values["schema_link_error"] = 1.0

        if not db_path:
            result.metric_values["execution_success"] = 1.0
            result.metric_values["result_accuracy"] = 1.0
            result.metric_values["result_f1"] = 1.0
            result.final_success = True
            return result

        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(sql)
                rows = cur.fetchall()
            result.metric_values["execution_success"] = 1.0

            # Resolve gold rows: from sample data or execute gold_sql
            if gold_result_rows is None and gold_sql:
                try:
                    with sqlite3.connect(db_path) as conn:
                        cur = conn.cursor()
                        cur.execute(gold_sql)
                        gold_result_rows = cur.fetchall()
                except Exception:
                    pass

            if gold_result_rows is not None:
                f1 = _compute_result_f1(rows, gold_result_rows)
                result.metric_values["result_f1"] = f1
                result.metric_values["result_accuracy"] = 1.0 if f1 >= 0.8 else 0.0
            elif gold_result_hash:
                pred_hash = _row_hash(rows) if rows else ""
                hash_match = pred_hash == gold_result_hash
                result.metric_values["result_accuracy"] = 1.0 if hash_match else 0.0
                result.metric_values["result_f1"] = 1.0 if hash_match else 0.0
            else:
                has_rows = rows is not None and len(rows) > 0
                result.metric_values["result_accuracy"] = 1.0 if has_rows else 0.0
                result.metric_values["result_f1"] = 1.0 if has_rows else 0.0

            result.final_success = result.metric_values["result_accuracy"] > 0
        except Exception as e:
            result.metric_values["execution_success"] = 0.0
            result.metric_values["result_accuracy"] = 0.0
            result.metric_values["result_f1"] = 0.0
            result.final_success = False
            result.failure_stage = "sql_execution"
            result.metadata["error"] = str(e)
        return result
