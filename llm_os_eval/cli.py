import typer
from pathlib import Path
from llm_os_eval.reporters.summary import summarize_jsonl

app = typer.Typer()

@app.command()
def hello():
    print("llm-os-eval-core ready")

@app.command()
def summarize(result_path: str):
    print(summarize_jsonl(Path(result_path)))

if __name__ == "__main__":
    app()
