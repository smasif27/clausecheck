"""Evaluate ClauseCheck's /api/ask against eval_set.json (standard library only).

    python eval/run_evals.py --file messy_contract.pdf
Needs the server (and Redis/DB) running and the PDF already uploaded and 'completed'.
Metrics: retrieval hit (are the evidence snippets inside the retrieved clauses?) and
answer correctness (key facts present, forbidden text absent, refusals where required).
"""
import argparse, json, time, urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REFUSALS = ["cannot determine", "can't determine", "not specified", "does not specify",
            "not stated", "not mentioned", "does not mention", "not defined", "no information",
            "does not contain", "doesn't contain", "does not include", "no mention",
            "does not address", "not provided", "no such clause", "no clause"]


def norm(t):
    t = t.lower()
    for a, b in {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
                 "\u2011": "-", "\u2013": "-", "\u2014": "-"}.items():
        t = t.replace(a, b)
    return " ".join(t.split())


def call(base, path, payload=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def find_document(base, filename):
    docs = [d for d in call(base, "/api/documents") if d["status"] == "completed"]
    match = [d for d in docs if d["filename"] == filename]
    if not match:
        raise SystemExit(f"No completed document named {filename!r}. Upload it in the UI first.")
    return match[0]["document_id"]


def score(item, answer, sources):
    a = norm(answer)
    res = {}
    if item["evidence"]:
        retrieved = norm(" ".join(s["text"] for s in sources))
        missing = [e for e in item["evidence"] if norm(e) not in retrieved]
        res["retrieval_hit"] = not missing
        res["missing_evidence"] = missing
    if item.get("must_refuse"):
        res["answer_correct"] = any(m in a for m in REFUSALS)
        res["missing_facts"] = [] if res["answer_correct"] else ["(should have refused)"]
    else:
        missing = [f for f in item["key_facts"] if not any(alt in a for alt in f.split("|"))]
        forbidden = [w for w in item.get("must_not", []) if w in a]
        res["answer_correct"] = not missing and not forbidden
        res["missing_facts"] = missing
        res["forbidden_found"] = forbidden
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--file", default="messy_contract.pdf", help="filename of an uploaded, completed document")
    ap.add_argument("--document-id")
    ap.add_argument("--eval-set", default=str(Path(__file__).with_name("eval_set.json")))
    ap.add_argument("--lambda-param", type=float, default=0.7)
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between questions (API rate limits)")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    items = json.load(open(args.eval_set, encoding="utf-8"))["questions"]
    doc_id = args.document_id or find_document(args.base_url, args.file)
    done, results = {}, []

    for item in items:
        history, cur = [], item.get("follow_up_of")
        while cur and cur in done and len(history) < 3:
            history.insert(0, {"question": done[cur]["question"], "answer": done[cur]["answer"]})
            cur = done[cur].get("follow_up_of")
        try:
            r = call(args.base_url, "/api/ask", {"question": item["question"], "document_id": doc_id,
                                                  "history": history, "lambda_param": args.lambda_param})
            answer, sources, err = r["answer"], r["sources"], None
        except Exception as e:
            answer, sources, err = "", [], str(e)
        res = score(item, answer, sources) if not err else {"answer_correct": False, "error": err}
        row = {**item, "answer": answer, **res, "retrieved": [x["text"][:80] for x in sources]}
        done[item["id"]] = row
        results.append(row)
        mark = "PASS" if row["answer_correct"] and row.get("retrieval_hit", True) else "FAIL"
        print(f"[{mark}] {item['id']} ({item['type']}) {item['question']}")
        time.sleep(args.sleep)

    by = defaultdict(lambda: [0, 0])
    for r in results:
        by[r["type"]][0] += r["answer_correct"]; by[r["type"]][1] += 1
    ret = [r["retrieval_hit"] for r in results if "retrieval_hit" in r]
    ok = sum(r["answer_correct"] for r in results)
    print("\n" + "=" * 60)
    print(f"Answer correctness : {ok}/{len(results)} = {ok / len(results):.0%}")
    if ret:
        print(f"Retrieval hit rate : {sum(ret)}/{len(ret)} = {sum(ret) / len(ret):.0%}")
    for t, (c, n) in by.items():
        print(f"  {t:<13} {c}/{n}")
    for r in results:
        if not r["answer_correct"] or r.get("retrieval_hit") is False:
            print(f"\n- {r['id']} {r['question']}\n  answer: {r['answer'][:200]!r}")
            for k in ("error", "missing_facts", "forbidden_found", "missing_evidence"):
                if r.get(k):
                    print(f"  {k}: {r[k]}")
    json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"), "lambda": args.lambda_param,
               "document_id": doc_id, "results": results}, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
