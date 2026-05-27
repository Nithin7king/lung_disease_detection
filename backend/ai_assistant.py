"""
PneumoScan AI Assistant — RAG + OpenRouter
==========================================
Architecture:
  1. All patient records → plain-text documents
  2. TF-IDF retrieval of the most relevant records for the question
  3. Build a rich context: aggregate stats + retrieved patient docs
  4. Send everything to OpenRouter → get answer + optional chart JSON
  5. Parse the structured response and return to the Flask endpoint

Falls back to simple rule-based analytics if OpenRouter is unavailable.
"""
from __future__ import annotations

import json
import os
import re
import requests
from typing import Dict, List, Optional, Tuple

# ── OpenRouter configuration ─────────────────────────────────────────

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Set via: export OPENROUTER_API_KEY="your_real_key"
OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_MODELS: List[str] = [
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free",
    "google/gemma-4-26b-a4b-it:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
]

TIMEOUT: int = 120


# ── OpenRouter call (with model fallback) ────────────────────────────

def _call_openrouter(messages: list) -> Optional[str]:
    """Try each model in OPENROUTER_MODELS until one succeeds."""
    if not OPENROUTER_API_KEY:
        print("[AI] No OpenRouter API key set.")
        return None

    for model in OPENROUTER_MODELS:
        try:
            print(f"[AI] Trying model: {model}")
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "PneumoScan AI",
                },
                data=json.dumps({
                    "model": model,
                    "messages": messages,
                    "max_tokens": 800,
                    "temperature": 0.7,
                }),
                timeout=TIMEOUT,
            )

            print(f"[AI] Status: {response.status_code}")
            print(response.text)

            if response.status_code != 200:
                print(f"[AI] Non-200 from {model}, trying next.")
                continue

            data = response.json()

            if "error" in data:
                print(f"[AI] Error in response from {model}: {data['error']}")
                continue

            choices = data.get("choices", [])
            if not choices:
                print(f"[AI] No choices from {model}, trying next.")
                continue

            message = choices[0].get("message", {})
            content = message.get("content", "")

            # Gemini sometimes returns list format
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )

            if content and content.strip():
                print(f"[AI] Success with model: {model}")
                return content.strip()

        except requests.exceptions.Timeout:
            print(f"[AI] Timeout on model: {model}")
        except Exception as exc:
            print(f"[AI] Error with model {model}: {exc}")

    print("[AI] All models exhausted.")
    return None


# ── Text-cleaning helpers ────────────────────────────────────────────

def clean_disease(raw: str) -> str:
    """Extract bare disease name from the verbose model prediction string."""
    s = str(raw or "")
    s = re.sub(r"respiratory\s+disorder\s+detected\s*:?", "", s, flags=re.I)
    s = re.sub(r"predicted?\s+disorder\s+detected\s*:?", "", s, flags=re.I)
    s = re.sub(r"prediction\s*:?", "", s, flags=re.I)
    s = re.sub(r"with\s+probability", "", s, flags=re.I)
    s = re.sub(r"[\d.]+\s*%", "", s)
    s = re.sub(r"[,;]?\s*second.*", "", s, flags=re.I)
    return s.strip()


def extract_probability(raw: str) -> float:
    """Extract probability percentage from the prediction string."""
    m = re.search(r"([\d.]+)\s*%", str(raw or ""))
    if not m:
        return 0.0
    val = float(m.group(1))
    return min(val if val > 1 else val * 100, 100.0)


# ── Process records ──────────────────────────────────────────────────

def process_records(firestore_records: list) -> List[dict]:
    """Convert raw Firestore dicts into clean patient dicts."""
    result = []
    for r in firestore_records:
        pred = r.get("prediction", "") or ""
        result.append({
            "patient_name": r.get("patient_name", "Unknown"),
            "disease": clean_disease(pred),
            "probability": round(extract_probability(pred), 2),
            "date": r.get("date", ""),
            "_raw_prediction": pred,
        })
    return result


# ── Document creation (for RAG) ──────────────────────────────────────

def _records_to_docs(records: List[dict]) -> List[str]:
    """Convert each patient record to a searchable text document."""
    return [
        f"Record {i}: Patient={r['patient_name']}, "
        f"Disease={r['disease']}, "
        f"Probability={r['probability']:.1f}%, "
        f"Date={r['date']}"
        for i, r in enumerate(records, 1)
    ]


# ── TF-IDF retrieval ─────────────────────────────────────────────────

def _retrieve_relevant(
    question: str,
    docs: List[str],
    top_k: int = 20,
) -> List[str]:
    """Return the top_k most relevant documents using TF-IDF cosine similarity."""
    if not docs:
        return []
    if len(docs) <= top_k:
        return docs

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        corpus = docs + [question]
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
        mat = vec.fit_transform(corpus)
        sims = cosine_similarity(mat[-1], mat[:-1])[0]
        top_idx = np.argsort(sims)[-top_k:][::-1]
        return [docs[i] for i in top_idx]
    except Exception:
        return docs[:top_k]


# ── Aggregate statistics ─────────────────────────────────────────────

def _compute_stats(records: List[dict]) -> dict:
    """Compute aggregate analytics over all records."""
    n = len(records)
    disease_counts: Dict[str, int] = {}
    prob_by_disease: Dict[str, List[float]] = {}

    for r in records:
        d = r["disease"]
        disease_counts[d] = disease_counts.get(d, 0) + 1
        prob_by_disease.setdefault(d, []).append(r["probability"])

    avg_prob_by_disease = {
        d: round(sum(v) / len(v), 1) for d, v in prob_by_disease.items()
    }
    overall_avg = round(sum(r["probability"] for r in records) / max(n, 1), 1)
    top_disease = (
        max(disease_counts, key=lambda k: disease_counts[k])
        if disease_counts else "N/A"
    )

    return {
        "total_patients": n,
        "disease_counts": disease_counts,
        "avg_prob_by_disease": avg_prob_by_disease,
        "overall_avg_probability": overall_avg,
        "most_common_disease": top_disease,
        "sorted_diseases": sorted(disease_counts.items(), key=lambda x: -x[1]),
        # kept for _parse_ai_response auto-chart
        "distribution": disease_counts,
    }


# ── System prompt ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are PneumoScan AI, an intelligent medical assistant embedded in a \
respiratory disease detection web application. Doctors can ask you anything \
about their patients — statistics, specific patients, trends, advice, \
recommendations, or general respiratory medicine questions.

You are given:
1. Aggregate statistics for all of the doctor's patients.
2. The most relevant individual patient records (retrieved via RAG).
3. The doctor's question.

Instructions:
- Answer the question accurately and helpfully using the provided data.
- Use markdown formatting (bold headings, lists, etc.) for clarity.
- Be concise but complete.
- If the question needs a chart, include a JSON block at the END of your \
response with this exact format (nothing else after it):
```chart
{"type": "pie", "data": {"Disease A": 5, "Disease B": 3}}
```
  Use "pie" for distributions, "bar" for comparisons / averages.
- If no chart is needed, do NOT include a ```chart block at all.
- Never reveal that you are using any external AI or model.
- If asked about general respiratory medicine, answer professionally.
"""


# ── OpenRouter RAG answering ─────────────────────────────────────────

def _openrouter_rag_answer(
    question: str,
    records: List[dict],
    stats: dict,
    relevant_docs: List[str],
) -> Optional[Tuple[str, Optional[dict]]]:
    """Build a RAG prompt and call OpenRouter."""

    avg_d = stats["avg_prob_by_disease"]
    stats_text = (
        f"Total patients: {stats['total_patients']}\n"
        f"Most common disease: {stats['most_common_disease']}\n"
        f"Overall average probability: {stats['overall_avg_probability']:.1f}%\n"
        "Disease distribution:\n"
        + "\n".join(
            f"  - {d}: {c} patient(s), avg prob {avg_d.get(d, 0):.1f}%"
            for d, c in stats["sorted_diseases"]
        )
    )

    rag_context = (
        "\n".join(relevant_docs) if relevant_docs
        else "No specific records retrieved."
    )

    user_prompt = (
        f"=== AGGREGATE STATISTICS ===\n{stats_text}\n\n"
        f"=== RELEVANT PATIENT RECORDS (RAG) ===\n{rag_context}\n\n"
        f"=== DOCTOR'S QUESTION ===\n{question}"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    raw_text = _call_openrouter(messages)
    if raw_text is None:
        return None

    return _parse_ai_response(raw_text, stats)


# ── Response parser ──────────────────────────────────────────────────

def _parse_ai_response(
    raw: str,
    stats: dict,
) -> Tuple[str, Optional[dict]]:
    """Split the AI response into (answer_text, chart_dict)."""
    chart_pattern = re.search(
        r"```chart\s*\n([\s\S]*?)\n```", raw, flags=re.I
    )
    chart: Optional[dict] = None

    if chart_pattern:
        try:
            parsed = json.loads(chart_pattern.group(1).strip())
            if (
                isinstance(parsed.get("data"), dict)
                and parsed.get("type") in ("pie", "bar", "line")
            ):
                chart = {"type": parsed["type"], "data": parsed["data"]}
        except Exception:
            pass
        answer = raw[: chart_pattern.start()].strip()
    else:
        answer = raw.strip()

    # Auto-generate a pie chart if the answer discusses distribution
    if chart is None and _answer_needs_chart(answer) and stats.get("distribution"):
        chart = {
            "type": "pie",
            "data": stats["distribution"],
        }

    return answer, chart


def _answer_needs_chart(answer: str) -> bool:
    """Heuristic: does the answer mention distribution / counts prominently?"""
    lower = answer.lower()
    signals = [
        "distribution", "breakdown", "most common",
        "percentage", "pie", "chart", "graph", "proportion",
    ]
    return sum(1 for s in signals if s in lower) >= 2


# ── Rule-based fallback ──────────────────────────────────────────────

_DISEASE_KEYS = [
    "copd", "asthma", "pneumonia", "bronchiectasis",
    "bronchiolitis", "healthy", "normal", "urti", "lrti",
    "crackle", "wheez",
]


def _disease_mention(text: str) -> Optional[str]:
    t = text.lower()
    for key in _DISEASE_KEYS:
        if key in t:
            return key
    return None


def _rule_based_answer(
    question: str,
    records: List[dict],
    stats: dict,
) -> Tuple[str, Optional[dict]]:
    """Simple keyword-based answers used when OpenRouter is unavailable."""
    q       = question.lower()
    n       = stats["total_patients"]
    dc      = stats["disease_counts"]
    top     = stats["most_common_disease"]
    avg     = stats["overall_avg_probability"]
    sorted_d = stats["sorted_diseases"]

    pie_chart = (
        {"type": "pie", "data": {d: c for d, c in sorted_d}}
        if dc else None
    )

    # ── Count queries ────────────────────────────────────────────────
    if any(kw in q for kw in ["how many", "count", "total", "number"]):
        dis = _disease_mention(q)
        if dis:
            c = sum(1 for r in records if dis in r["disease"].lower())
            return (
                f"You have **{c}** {dis.upper()} patient{'s' if c != 1 else ''} "
                f"out of **{n}** total.",
                {"type": "bar", "data": {dis.upper(): c, "Others": n - c}},
            )
        return f"You have **{n}** total patient{'s' if n != 1 else ''}.", pie_chart

    # ── Distribution ─────────────────────────────────────────────────
    if any(kw in q for kw in ["distribution", "breakdown", "most common", "diseases"]):
        rows = "\n".join(
            f"- **{d}**: {c} ({c / n * 100:.1f}%)" for d, c in sorted_d
        )
        return f"**Disease distribution** ({n} patients):\n\n{rows}", pie_chart

    # ── Probability / average ────────────────────────────────────────
    if any(kw in q for kw in ["average", "probability", "score"]):
        rows = "\n".join(
            f"- **{d}**: {a:.1f}%" for d, a in stats["avg_prob_by_disease"].items()
        )
        return (
            f"Average probability: **{avg:.1f}%**\n\nBy disease:\n{rows}",
            {"type": "bar", "data": stats["avg_prob_by_disease"]},
        )

    # ── List patients for a disease ──────────────────────────────────
    dis = _disease_mention(q)
    if dis and any(kw in q for kw in ["show", "list", "who", "patient"]):
        f_recs = [r for r in records if dis in r["disease"].lower()]
        if not f_recs:
            return f"No patients with **{dis.upper()}** found.", None
        rows = "\n".join(
            f"{i}. **{r['patient_name']}** — {r['probability']:.1f}% ({r['date']})"
            for i, r in enumerate(f_recs, 1)
        )
        return f"**{len(f_recs)} {dis.upper()} patients:**\n\n{rows}", None

    # ── Recent patients ──────────────────────────────────────────────
    if any(kw in q for kw in ["recent", "latest", "last"]):
        top5 = sorted(records, key=lambda r: r["date"] or "", reverse=True)[:5]
        rows = "\n".join(
            f"- **{r['patient_name']}** — {r['disease']} ({r['probability']:.1f}%) on {r['date']}"
            for r in top5
        )
        return f"**{len(top5)} most recent patients:**\n\n{rows}", None

    # ── Summary / overview ───────────────────────────────────────────
    if any(kw in q for kw in ["summary", "overview", "report"]):
        rows = "\n".join(f"  - {d}: {c}" for d, c in sorted_d)
        return (
            f"**Patient Summary**\n\n"
            f"- Total: **{n}**\n"
            f"- Most common: **{top}**\n"
            f"- Avg probability: **{avg:.1f}%**\n\n"
            f"**Disease breakdown:**\n{rows}",
            pie_chart,
        )

    # ── Precautions / advice ─────────────────────────────────────────
    if any(kw in q for kw in [
        "precaution", "advice", "recommend", "treatment",
        "prevent", "care", "manage",
    ]):
        dis = _disease_mention(q)
        precautions = {
            "copd": [
                "Avoid smoking and secondhand smoke exposure.",
                "Use prescribed inhalers (bronchodilators/steroids) regularly.",
                "Get annual flu and pneumococcal vaccines.",
                "Practice breathing exercises (pursed-lip breathing).",
                "Avoid air pollutants and dust.",
                "Maintain a healthy weight and stay active with light exercise.",
            ],
            "asthma": [
                "Identify and avoid personal triggers (dust, pollen, cold air, exercise).",
                "Always carry a rescue inhaler.",
                "Follow an asthma action plan prescribed by your doctor.",
                "Keep indoor air clean — use air purifiers if needed.",
                "Monitor peak flow readings regularly.",
            ],
            "pneumonia": [
                "Complete the full course of prescribed antibiotics.",
                "Rest adequately and stay well hydrated.",
                "Get the pneumococcal and flu vaccines.",
                "Avoid smoking — it impairs lung recovery.",
                "Follow up with chest X-ray to confirm resolution.",
                "Seek immediate care if breathing worsens.",
            ],
            "bronchiectasis": [
                "Perform airway clearance techniques (chest physiotherapy) daily.",
                "Stay well hydrated to thin mucus secretions.",
                "Take prescribed antibiotics during exacerbations promptly.",
                "Avoid respiratory infections — practice good hand hygiene.",
                "Get vaccinated against flu and pneumococcus.",
            ],
            "bronchiolitis": [
                "Ensure adequate hydration and rest.",
                "Use saline nasal drops to ease congestion.",
                "Monitor oxygen levels — seek care if below 95%.",
                "Avoid exposure to cigarette smoke.",
                "Hospitalisation may be needed for infants with severe symptoms.",
            ],
            "urti": [
                "Rest and drink plenty of fluids.",
                "Use saline nasal rinses for congestion relief.",
                "Avoid antibiotics unless bacterial infection is confirmed.",
                "Practise good hand hygiene to prevent spread.",
                "Use steam inhalation to soothe airways.",
            ],
            "lrti": [
                "Complete the full antibiotic course if prescribed.",
                "Rest and maintain good hydration.",
                "Use a humidifier to ease breathing.",
                "Monitor for worsening symptoms — seek urgent care if needed.",
                "Avoid smoking and pollutants during recovery.",
            ],
            "crackle": [
                "Investigate underlying cause (heart failure, fibrosis, pneumonia).",
                "Follow treatment plan specific to the underlying condition.",
                "Avoid respiratory irritants.",
                "Schedule regular follow-up spirometry and imaging.",
            ],
            "wheez": [
                "Identify and avoid allergens and irritants.",
                "Use bronchodilator inhalers as prescribed.",
                "Keep a symptom diary to track triggers.",
                "Maintain a clean, dust-free living environment.",
                "Seek emergency care if wheezing is severe and unrelieved.",
            ],
        }

        if dis and dis in precautions:
            steps = "\n".join(f"{i}. {p}" for i, p in enumerate(precautions[dis], 1))
            return (
                f"**Precautions & Recommendations for {dis.upper()}:**\n\n{steps}",
                None,
            )

        return (
            "**General Respiratory Precautions:**\n\n"
            "1. Avoid smoking and exposure to secondhand smoke.\n"
            "2. Get annual flu and pneumococcal vaccinations.\n"
            "3. Practise good hand hygiene to reduce infection risk.\n"
            "4. Stay well hydrated and maintain a healthy diet.\n"
            "5. Exercise regularly to strengthen respiratory muscles.\n"
            "6. Avoid known allergens and air pollutants.\n"
            "7. Follow prescribed medication plans and attend regular check-ups.\n"
            "8. Seek early medical attention for worsening symptoms.",
            None,
        )

    # ── Generic fallback ─────────────────────────────────────────────
    return (
        f"I found **{n}** patient{'s' if n != 1 else ''}. "
        f"Most common: **{top}**. Average probability: **{avg:.1f}%**.\n\n"
        "*(AI is currently unavailable — check your OPENROUTER_API_KEY and server logs.)*",
        pie_chart,
    )


# ── Master entry point ───────────────────────────────────────────────

def smart_answer(question: str, records: List[dict]) -> Tuple[str, Optional[dict]]:
    """Main entry point called by the Flask endpoint."""
    if not records:
        return (
            "No patient records found for your account. "
            "Run some analyses first and they will appear here.",
            None,
        )

    stats        = _compute_stats(records)
    docs         = _records_to_docs(records)
    relevant_docs = _retrieve_relevant(question, docs, top_k=min(25, len(records)))

    # Try OpenRouter (RAG)
    try:
        result = _openrouter_rag_answer(question, records, stats, relevant_docs)
        if result:
            return result
    except Exception as exc:
        print(f"[AI] OpenRouter RAG failed: {exc}")

    # Fallback: rule-based analytics
    return _rule_based_answer(question, records, stats)