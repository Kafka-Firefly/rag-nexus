"""Run Ragas metrics on a hand-labeled JSONL evaluation set against the live API."""
import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness


async def ask_api(client: httpx.AsyncClient, question: str) -> tuple[str, list[str]]:
    answer, citations = [], []
    async with client.stream("POST", "/chat", json={"question": question}, timeout=180) as response:
        response.raise_for_status()
        event_name = "message"
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                if event_name == "citations":
                    citations = [item["text"] for item in payload]
                elif event_name == "error":
                    raise RuntimeError(payload.get("detail", "Chat API returned an error"))
                elif event_name == "message" and "token" in payload:
                    answer.append(payload["token"])
            elif not line:
                event_name = "message"
    return "".join(answer), citations


async def run(args):
    load_dotenv(Path(__file__).with_name(".env"))
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Evaluation JSONL contains no rows")
    for index, row in enumerate(rows, 1):
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("question", "reference")):
            raise ValueError(f"Row {index} requires non-empty string fields: question, reference")

    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("LLM_API_KEY", "") or "ollama"
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=90, max_retries=2)
    evaluator_llm = llm_factory(llm_model, client=client)
    evaluator_embeddings = embedding_factory("openai", model=os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-3-small"), client=client)
    metrics = {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }
    api = httpx.AsyncClient(base_url=args.api_url.rstrip("/"))
    results = []
    try:
        for row in rows:
            answer, contexts = await ask_api(api, row["question"])
            scores = {}
            for name, metric in metrics.items():
                score = await metric.ascore(user_input=row["question"], response=answer,
                    retrieved_contexts=contexts, reference=row["reference"])
                scores[name] = score.value
            results.append({"question": row["question"], "answer": answer,
                "retrieved_contexts": contexts, "scores": scores})
    finally:
        await api.aclose()
        await client.close()

    averages = {name: sum(item["scores"][name] for item in results) / len(results) for name in metrics}
    report = {"sample_count": len(results), "mean_scores": averages, "results": results,
        "note": "LLM-judged scores vary by evaluator model; use a domain-reviewed dataset and compare runs using the same evaluator."}
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UTF-8 JSONL with question and reference fields")
    parser.add_argument("--output", help="Optional report JSON destination")
    parser.add_argument("--api-url", default=os.getenv("RAG_API_URL", "http://localhost:8000"))
    asyncio.run(run(parser.parse_args()))
