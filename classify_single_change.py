

import sys
import json
import argparse


from llm_bench_utils_v5 import (
    call_llm_raw, extract_last_json, sanitise, get_feature_cols, format_features,
    MSG_COL, _ERROR_RESULT,
)
from prompt_zeroshot_v5 import SYSTEM_PROMPT, build_prompt

from fetch_single_change import fetch_single_change_row

#MODEL = "z-ai/glm-5.2"
#PROVIDER = "nvidia"
MODEL = "zai-org/GLM-5.2"
PROVIDER = "baseten"
#MODEL = "gpt-4o-mini"  
#PROVIDER = "openai"

LLM_TIMEOUT_SECONDS = 180   # per-call timeout
LLM_MAX_ATTEMPTS = 4      

RESOLUTION_SYSTEM_PROMPT = """\
You are helping resolve a divergent Gerrit code review discussion from
OpenStack. You will be given the divergence type, the specific evidence
quote showing the conflicting positions, and the FULL discussion thread
(which may include messages from later patchsets, after the divergence
was raised).

IMPORTANT - do not confuse patchsets with related changes: a patchset is a
revision of THIS SAME change (small numbers like 1, 2, 3, "latest"). A
different change number (e.g. a much larger number referenced as
superseding, abandoning, or replacing this one) is a SEPARATE Gerrit
change, not a patchset. Never write "patchset <large number>" or
"PS<large number>" - if you are referring to a different change, call it
"change <number>", never a patchset of this one.

First check: has this specific disagreement ALREADY been fully addressed
or resolved LATER in the discussion (e.g. author made the change, reviewer
explicitly acknowledged/approved it, or a follow-up patchset fixed it)?

If YES, already resolved - do not invent recommendations. Reply with:
{"already_resolved": true, "abandoned": false, "resolution_summary": "<one sentence on how/where it was resolved, max 25 words>", "recommendations": []}

Otherwise, check: was the change ABANDONED (by owner or a core) before the
disagreement was actually settled - i.e. the discussion just stopped rather
than being resolved? This is NOT the same as resolution. If YES, still
propose exactly THREE distinct, actionable recommendations framed as what
should happen IF the change were reconsidered/restored. Reply with:
{"already_resolved": false, "abandoned": true, "resolution_summary": "<one sentence on when/why it was abandoned, max 25 words>", "recommendations": ["<rec 1, max 25 words>", "<rec 2>", "<rec 3>"]}

If NO to both - still open/unresolved as of the latest message - propose
exactly THREE distinct, actionable ways to resolve it. Be concrete - name
what should happen next (e.g. "escalate to a core reviewer", "author
should justify the approach in the commit message", "align with the
release schedule before proceeding"). Reply with:
{"already_resolved": false, "abandoned": false, "resolution_summary": "", "recommendations": ["<rec 1, max 25 words>", "<rec 2>", "<rec 3>"]}

Reply with ONLY the JSON object, no markdown, no text before or after.
"""

def _safe_call_llm(system_prompt: str, user_prompt: str, num_predict: int) -> tuple:
    """Wraps call_llm_raw with a hard timeout and retries. On an empty
    response (most often NIM 429 rate limiting), waits 30s before
    retrying, since NIM's cooldown window is longer than a quick retry."""
    import concurrent.futures
    import time

    RATE_LIMIT_BACKOFF_SECONDS = 30

    last_error = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    call_llm_raw,
                    model=MODEL,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    num_predict=num_predict,
                    temperature=0.0,
                    seed=42,
                    provider=PROVIDER,
                )
                raw = future.result(timeout=LLM_TIMEOUT_SECONDS)
            if raw and raw.strip():
                return raw, None
            last_error = "LLM returned an empty response (likely rate-limited)"
        except concurrent.futures.TimeoutError:
            last_error = f"LLM call timed out after {LLM_TIMEOUT_SECONDS}s"
        except Exception as exc:
            last_error = f"LLM call failed: {exc}"

        if attempt < LLM_MAX_ATTEMPTS:
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)

    return "", last_error

def classify_change(row: dict) -> dict:
    """Run the locked zeroshot v5 classifier on one row.
    Explicitly handles: empty discussion, and LLM/API failure."""
    msg_text = (row.get(MSG_COL) or "").strip()
    votes_text = (row.get("all_votes") or "").strip()

    # ── Empty discussion: no messages AND no vote signal at all ──
    if not msg_text and not votes_text:
        return {
            "divergence": "non-divergent",
            "type": "non-divergent",
            "type_category": "non-divergent",
            "reason": "Empty discussion - no review messages or vote signals to evaluate",
            "evidence": "",
            "confidence": "low",
            "divergent_patchsets": "",
            "skipped": True,
        }

    feature_cols = get_feature_cols(row.keys())
    features_block = format_features(row, feature_cols)
    user_prompt = build_prompt(msg_text, features_block)

    raw, error = _safe_call_llm(SYSTEM_PROMPT, user_prompt, num_predict=800)

    if error:
        result = dict(_ERROR_RESULT)
        result["reason"] = error
        result["skipped"] = False
        return result

    parsed = extract_last_json(raw)
    if parsed is not None:
        result = sanitise(parsed)
        result["skipped"] = False
        return result

    result = dict(_ERROR_RESULT)
    result["reason"] = "Could not parse LLM output as valid JSON"
    result["raw_output"] = raw
    result["skipped"] = False
    return result

def recommend_resolutions(divergence_type: str, evidence: str, full_discussion: str) -> dict:
    user_prompt = (
        f"Divergence type: {divergence_type}\n"
        f"Evidence (where the disagreement was first raised): \"{evidence}\"\n\n"
        f"Full discussion thread:\n{full_discussion}\n\n"
        "Assessment:"
    )
    raw, error = _safe_call_llm(RESOLUTION_SYSTEM_PROMPT, user_prompt, num_predict=300)

    empty = {"already_resolved": False, "abandoned": False, "resolution_summary": "", "recommendations": []}
    if error:
        empty["recommendations"] = [f"Could not generate recommendations: {error}"]
        return empty
    parsed = extract_last_json(raw)



    if isinstance(parsed, dict):
        recs = parsed.get("recommendations", [])
        recs = [str(r).strip() for r in recs if str(r).strip()] if isinstance(recs, list) else []
        return {
            "already_resolved": bool(parsed.get("already_resolved", False)),
            "abandoned": bool(parsed.get("abandoned", False)),
            "resolution_summary": str(parsed.get("resolution_summary", "")).strip(),
            "recommendations": recs[:3],
        }
    empty["recommendations"] = [raw.strip()[:500] or "No recommendation returned."]

    return empty

def check_divergence(change_number) -> dict:
    """Full pipeline for one change: fetch -> classify -> (if divergent) recommend.
    Never raises for expected failure modes (empty discussion, LLM timeout,
    fetch failure) - always returns a dict the caller can display."""
    try:
        row = fetch_single_change_row(change_number)
    except Exception as exc:
        return {
            "change_number": change_number,
            "subject": "",
            "project": "",
            "divergence": "error",
            "type_category": "error",
            "evidence": "",
            "divergent_patchsets": "",
            "recommendations": [],
            "error": f"Could not fetch change from Gerrit: {exc}",
        }

    result = classify_change(row)
    result["recommendations"] = []
    result["already_resolved"] = False
    result["resolution_summary"] = ""
    if result["divergence"] == "divergent":
        res = recommend_resolutions(
            result["type_category"], result["evidence"], row.get(MSG_COL, "")
        )
        result["already_resolved"] = res["already_resolved"]
        result["abandoned"] = res["abandoned"]
        result["resolution_summary"] = res["resolution_summary"]
        result["recommendations"] = res["recommendations"]


    

    result["change_number"] = row["change_number"]
    result["subject"] = row["subject"]
    result["project"] = row["project"]
    result["patchset_count"] = row["patchset_count"]

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("change_number", help="Gerrit change number, e.g. 123456")
    args = parser.parse_args()

    result = check_divergence(args.change_number)

    print(f"\nChange {result['change_number']}: {result.get('subject', '')}")
    print(f"Project: {result.get('project', '')}\n")

    if result["divergence"] == "error":
        print(f"Error: {result.get('error') or result.get('reason', 'unknown error')}")
    elif result.get("skipped"):
        print("Divergence: No (empty discussion, skipped LLM call)")
    elif result["divergence"] == "divergent":
        print("Divergence: Yes")
        print(f"Type: {result['type_category']}")
        print(f"Patchsets: {result.get('divergent_patchsets', '')}")
        if result.get("already_resolved"):
            print(f"Already resolved: {result.get('resolution_summary', '')}")
        else:
            print("Recommendations:")
            for i, rec in enumerate(result["recommendations"], 1):
                print(f"  {i}. {rec}")

    else:
        print("Divergence: No")
