"""
llm_bench_utils_v5.py


"""



import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import openpyxl

try:
    import ollama
except ImportError:
    ollama = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


MSG_COL      = "all_messages"
VOTES_COL    = "all_votes"
MAX_RETRIES  = 2
RETRY_DELAY  = 6


EXCLUDE_COLS = {
    "manual_label", "manual_notes", "manual_decision",
    "change_number", "url",
    "updated", "review_duration_days",
    "top_level_messages", "inline_comments",
    "agreement_votes", "divergent_votes", "divergence_at_patchset_votes",
    "all_votes", "vote_timeline", "review_scores_detail","agreement","divergent","divergence_at_patchset",
}

_TYPE_CATEGORIES  = {"technical", "procedural", "coordination", "social"}
_VALID_CONFIDENCE = {"low", "medium", "high"}

_EMPTY_TYPE_MARKERS = {"", "none", "nan", "null", "unspecified", "other"}


_ERROR_RESULT = {
    "divergence":          "error",
    "type":                "error",
    "type_category":       "error",
    "reason":              "LLM call failed after retries - no output returned for this row",
    "evidence":            "",
    "confidence":          "low",
    "divergent_patchsets": "",
}

_DEFAULT_NO_SIGNAL = {
    "divergence":          "non-divergent",
    "type":                "non-divergent",
    "type_category":       "non-divergent",
    "reason":              "No review messages or vote signals",
    "evidence":            "",
    "confidence":          "low",
    "divergent_patchsets": "",
}


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────── COLUMNS / FEATURES ────────────────

def get_feature_cols(all_cols):
    return [c for c in all_cols if c not in EXCLUDE_COLS and c != MSG_COL]


def format_features(row: dict, feature_cols) -> str:
    """Render every non-empty feature column as a labeled bullet line.
    No value is ever truncated or capped here - including ci_cd_signals."""
    lines = []
    for col in feature_cols:
        val = (row.get(col) or "").strip()
        if not val or val.lower() in ("none", "nan", "null", "[]", "{}"):
            continue
        label = col.replace("_", " ")
        lines.append(f"- {label}: {val}")
    return "\n".join(lines) if lines else "(no additional metadata)"


# ─────────────────────────── PROVIDERS ─────────────────────────

# Any provider in this dict is served through the same OpenAI-compatible
# chat.completions call path (_call_openai_compatible). "openai" uses the
# SDK's default base_url (base_url=None). Add a new OpenAI-compatible host
# by adding one entry here - no other code needs to change.
PROVIDER_CONFIGS = {
    "openai": {
        "base_url":    None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "nvidia": {

        "base_url":    "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
    },
        "baseten": {
        "base_url":    "https://inference.baseten.co/v1",
        "api_key_env": "BASETEN_API_KEY",
    },
}


# OpenAI/NIM currently ignore "seed" for most models - kept as a no-op
# passthrough on the ollama path only.
def _call_ollama(model, system_prompt, user_prompt, num_predict, temperature,
                  seed, response_format, repeat_penalty):
    if ollama is None:
        raise RuntimeError("ollama package not installed (pip install ollama --break-system-packages)")
    options = {"temperature": temperature, "num_predict": num_predict}
    if seed is not None:
        options["seed"] = seed
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty
    kwargs = dict(model=model, prompt=user_prompt, system=system_prompt, options=options)
    if response_format:
        kwargs["format"] = response_format
    response = ollama.generate(**kwargs)
    return (response.get("response") or "").strip()


_CLIENTS = {}  # provider name -> cached OpenAI() client


def _get_client(provider: str):
    if provider not in PROVIDER_CONFIGS:
        raise RuntimeError(f"Unknown OpenAI-compatible provider '{provider}'. "
                            f"Known: {sorted(PROVIDER_CONFIGS)}")
    if provider not in _CLIENTS:
        if OpenAI is None:
            raise RuntimeError("openai package not installed (pip install openai --break-system-packages)")
        cfg = PROVIDER_CONFIGS[provider]
        api_key = os.environ.get(cfg["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{cfg['api_key_env']} is not set in the environment "
                                f"(required for --provider {provider})")
        client_kwargs = dict(api_key=api_key)
        if cfg["base_url"]:
            client_kwargs["base_url"] = cfg["base_url"]
        _CLIENTS[provider] = OpenAI(**client_kwargs)
    return _CLIENTS[provider]


def _call_openai_compatible(provider, model, system_prompt, user_prompt,
                             num_predict, temperature, response_format):
    client = _get_client(provider)
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=num_predict,
    )
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    try:
        completion = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Some OpenAI-compatible hosts (certain NIM-hosted models, e.g.
        # GLM) reject/ignore response_format={"type":"json_object"} and
        # raise instead of silently ignoring it. Retry once without it -
        # the prompt text itself still asks for JSON-only output, it just
        # won't be schema-enforced on this call.
        if "response_format" in kwargs:
            log(f"  [warn] {provider}:{model} rejected response_format "
                f"({exc}); retrying once without it")
            kwargs.pop("response_format")
            completion = client.chat.completions.create(**kwargs)
        else:
            raise
    return (completion.choices[0].message.content or "").strip()


def call_llm_raw(model: str, system_prompt: str, user_prompt: str,
                  num_predict: int = 400, temperature: float = 0.0,
                  seed: int = 42, response_format: str = None,
                  repeat_penalty: float = None, provider: str = "ollama") -> str:

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider in PROVIDER_CONFIGS:
                return _call_openai_compatible(provider, model, system_prompt, user_prompt,
                                                num_predict, temperature, response_format)
            return _call_ollama(model, system_prompt, user_prompt, num_predict,
                                 temperature, seed, response_format, repeat_penalty)
        except Exception as exc:
            wait = RETRY_DELAY
            if "429" in str(exc):
                wait = 30
            log(f"  [warn] LLM call attempt {attempt}/{MAX_RETRIES} failed ({provider}:{model}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
    return ""


def call_ollama_raw(*args, **kwargs):
    """Backward-compatible alias - always routes to the ollama provider."""
    kwargs.setdefault("provider", "ollama")
    return call_llm_raw(*args, **kwargs)


# ─────────────────────────── PARSING ───────────────────────────

def extract_last_json(raw: str):

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    decoder = json.JSONDecoder()
    candidates = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
            candidates.append(obj)
        except json.JSONDecodeError:
            continue

    if not candidates:
        return None

    schema_matches = [c for c in candidates if isinstance(c, dict) and "divergence" in c]
    if schema_matches:
        return schema_matches[-1]
    return candidates[-1]


def _clean_reason(text: str) -> str:

    if not text:
        return text
    t = text.strip()
    t = re.sub(r'^\[salvaged from[^\]]*\]\s*', '', t, flags=re.I)
    t = re.sub(r'(?im)^json\s*:\s*', '', t)
    t = re.sub(r'^```(?:json)?\s*', '', t)
    t = re.sub(r'\s*```$', '', t)
    return t.strip()


def _categorize_type(raw_type: str) -> str:

    t = (raw_type or "").strip().lower()
    if not t or t in _EMPTY_TYPE_MARKERS:
        return "error"
    if t in _TYPE_CATEGORIES:
        return t
    if "/" in t:
        parts = [p.strip() for p in t.split("/") if p.strip()]
        seen = []
        for p in parts:
            if p in _TYPE_CATEGORIES and p not in seen:
                seen.append(p)
        if seen:
            return "/".join(seen[:2])

    hits = [c for c in _TYPE_CATEGORIES if c in t]
    if hits:
        return "/".join(hits[:2])
    return "error"


def salvage_from_prose(raw: str) -> dict:
    """Second-tier parser for zeroshot/fewshot outputs when the model
    never produced a JSON block at all."""
    text = raw.strip()
    if not text:
        return dict(_ERROR_RESULT)

    div_match = re.search(r'(?im)^\s*"?divergence"?\s*[:\-]\s*"?(divergent|non-divergent)', text)
    type_match = re.search(r'(?im)^\s*"?type"?\s*[:\-]\s*"?([a-zA-Z][a-zA-Z/\- ]*?)"?\s*$', text)
    conf_match = re.search(r'(?im)^\s*"?confidence"?\s*[:\-]\s*"?(low|medium|high)', text)
    ps_match = re.search(r'(?im)^\s*"?divergent_patchsets"?\s*[:\-]\s*"?([0-9,\s]+|all)', text)

    if div_match:
        divergence = div_match.group(1).lower()
    elif type_match and type_match.group(1).strip().lower() not in ("non-divergent", "non divergent", ""):
        divergence = "divergent"
    else:
        divergence = "non-divergent"

    result = {
        "divergence": divergence,
        "reason":     _clean_reason(text[:600]),
        "confidence": conf_match.group(1).lower() if conf_match else "low",
    }

    if divergence == "divergent":
        raw_type = type_match.group(1).strip().lower() if type_match else ""
        result["type"] = raw_type or "error"
        result["type_category"] = _categorize_type(raw_type)
        result["evidence"] = ""
        result["divergent_patchsets"] = ps_match.group(1).replace(" ", "") if ps_match else ""
    else:
        result["type"] = "non-divergent"
        result["type_category"] = "non-divergent"
        result["evidence"] = ""
        result["divergent_patchsets"] = ""

    return result


def salvage_from_cot(raw: str) -> dict:
    """Salvage parser matched to prompt_cot_v5.py's required format
    (Participants / Positions / Conflict / Decision lines), used when no
    valid JSON block follows those lines."""
    text = raw.strip()
    if not text:
        return dict(_ERROR_RESULT)

    decision_match = re.search(
        r'(?im)^\s*decision\s*:\s*(divergent|non-divergent)\s*[,\-]?\s*(.*)$', text
    )
    conflict_match = re.search(r'(?im)^\s*conflict\s*:\s*(yes|no)\b(.*)$', text)

    if decision_match:
        divergence = decision_match.group(1).lower()
        type_tail = decision_match.group(2).strip()
    elif conflict_match:
        divergence = "divergent" if conflict_match.group(1).lower() == "yes" else "non-divergent"
        type_tail = ""
    else:
        divergence = "non-divergent"
        type_tail = ""

    conf_match = re.search(r'(?im)^\s*"?confidence"?\s*[:\-]\s*"?(low|medium|high)', text)

    result = {
        "divergence": divergence,
        "reason":     _clean_reason(text[:600]),
        "confidence": conf_match.group(1).lower() if conf_match else "low",
    }

    if divergence == "divergent":
        raw_type = re.sub(r'(?i)^type\s*[:\-]\s*', '', type_tail).strip().lower()
        result["type"] = raw_type or "error"
        result["type_category"] = _categorize_type(raw_type)

        quote_match = re.search(r'"([^"]{1,150})"', conflict_match.group(0)) if conflict_match else None
        result["evidence"] = quote_match.group(1) if quote_match else ""

        ps_match = re.search(r'(?im)^\s*"?divergent_patchsets"?\s*[:\-]\s*"?([0-9,\s]+|all)', text)
        result["divergent_patchsets"] = ps_match.group(1).replace(" ", "") if ps_match else ""
    else:
        result["type"] = "non-divergent"
        result["type_category"] = "non-divergent"
        result["evidence"] = ""
        result["divergent_patchsets"] = ""

    return result


def sanitise(parsed: dict) -> dict:
    div_raw = str(parsed.get("divergence", parsed.get("disagreement", "non-divergent"))).strip().lower()
    divergence = "divergent" if div_raw in ("divergent", "true", "yes", "1") else "non-divergent"

    reason = _clean_reason(str(parsed.get("reason", "")).strip())
    if not reason:
        reason = "Divergence detected" if divergence == "divergent" else "No divergence detected"

    evidence = str(parsed.get("evidence", "")).strip()

    confidence = str(parsed.get("confidence", "low")).strip().lower()
    if confidence not in _VALID_CONFIDENCE:
        confidence = "low"

    patchsets = str(parsed.get("divergent_patchsets", "")).strip()
    patchsets = re.sub(r"\s*,\s*", ",", patchsets)

    raw_type = str(parsed.get("type", "")).strip()

    if divergence == "non-divergent":
        dtype         = "non-divergent"
        type_category = "non-divergent"
        evidence      = ""
        patchsets     = ""
    else:
        low = raw_type.lower()
        dtype = low if raw_type and low not in ("non-divergent", "nan", "null") else "error"
        type_category = _categorize_type(raw_type)
        if not patchsets or patchsets.lower() in ("none", "nan", "null"):
            patchsets = ""

    return {
        "divergence":          divergence,
        "type":                dtype,
        "type_category":       type_category,
        "reason":              reason,
        "evidence":            evidence,
        "confidence":          confidence,
        "divergent_patchsets": patchsets,
    }


def parse_response(raw: str) -> dict:
    parsed = extract_last_json(raw)
    if parsed is not None:
        return sanitise(parsed)
    if raw and raw.strip():
        return salvage_from_prose(raw)
    return dict(_ERROR_RESULT)


# ─────────────────────────── ROW PROCESSING ────────────────────

def process_row(row, model, system_prompt, build_prompt_fn, feature_cols,
                 num_predict=400, parse_fn=None, seed=42, response_format=None,
                 repeat_penalty=None, provider="ollama"):
    parse_fn = parse_fn or parse_response

    msg_text   = (row.get(MSG_COL) or "").strip()
    votes_text = (row.get(VOTES_COL) or "").strip()

    if not msg_text and not votes_text:
        result = dict(_DEFAULT_NO_SIGNAL)
        for k, v in result.items():
            row[f"llm_{k}"] = v
        return row

    features_block = format_features(row, feature_cols)
    user_prompt = build_prompt_fn(msg_text, features_block)
    raw = call_llm_raw(model, system_prompt, user_prompt, num_predict=num_predict,
                        seed=seed, response_format=response_format,
                        repeat_penalty=repeat_penalty, provider=provider)
    result = parse_fn(raw)

    if (
        result.get("divergence") == "divergent"
        and (
            str(result.get("type_category", "")).strip().lower() == "error"
            or not str(result.get("evidence", "")).strip()
        )
    ):
        retry_prompt = (
            user_prompt
            + "\n\nNOTE: your previous answer marked this \"divergent\" but did not provide "
              "a usable taxonomy type or evidence quote. Reply again with the full "
              "response, ensuring \"type\" is one of the 4 taxonomy categories (or a "
              "valid two-category combo like \"technical/procedural\") and \"evidence\" "
              "contains the specific conflict quote."
        )
        raw_retry = call_llm_raw(model, system_prompt, retry_prompt, num_predict=num_predict,
                                  seed=seed, response_format=response_format,
                                  repeat_penalty=repeat_penalty, provider=provider)
        if raw_retry.strip():
            retry_result = parse_fn(raw_retry)
            if (retry_result.get("divergence") == "divergent"
                    and retry_result.get("type_category", "error") != "error"):
                result = retry_result
                raw = raw_retry

    for k, v in result.items():
        row[f"llm_{k}"] = v
    row["llm_raw_response"] = raw
    return row


# ─────────────────────────── I/O ───────────────────────────────

def load_rows(input_path):
    log(f"Reading {input_path} …")
    wb = openpyxl.load_workbook(input_path, read_only=True, data_only=True)
    ws = wb.active
    all_rows = list(ws.iter_rows(values_only=True))
    cols = [str(c) for c in all_rows[0]]
    rows = [dict(zip(cols, [str(v) if v is not None else "" for v in r]))
            for r in all_rows[1:]]
    wb.close()
    log(f"  {len(rows)} rows, {len(cols)} columns")
    return cols, rows


# ─────────────────────────── MAIN RUNNER ───────────────────────

def run_benchmark(script_name, system_prompt, build_prompt_fn, num_predict=400, parse_fn=None,
                   seed=42, response_format=None, repeat_penalty=None):
    """
    script_name should include the "_v5" tag (e.g. "zeroshot_v5") so
    output files are written as results_<script_name>_<model>.xlsx -
    keeps v5 results from colliding with v2/v3/v4 results_*.xlsx files
    in the same results directory.
    """
    parser = argparse.ArgumentParser(description=f"Divergence benchmark — {script_name} prompt")
    parser.add_argument("--input",   required=True, help="Input Excel file (gerrit_reviews_v3.xlsx)")
    parser.add_argument("--model",   required=True,
                         help="Model tag. Ollama tag (e.g. qwen2.5:7b), an OpenAI model "
                              "name (e.g. gpt-4o-mini), or an NVIDIA NIM model id "
                              "(e.g. z-ai/glm-5.2) depending on --provider")
    parser.add_argument("--provider", choices=["ollama", "openai", "nvidia","baseten"], default="ollama",
                         help="Which backend to call (default: ollama, local). "
                              "'openai' and 'nvidia' both call an OpenAI-compatible "
                              "chat.completions API - 'openai' needs OPENAI_API_KEY, "
                              "'nvidia' needs NVIDIA_API_KEY (get one at build.nvidia.com).")
    parser.add_argument("--output",  default=None, help="Output Excel path (default: auto-named)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel LLM workers (default: 2)")
    args = parser.parse_args()

    cols, rows = load_rows(args.input)
    if MSG_COL not in cols:
        sys.exit(f"Column '{MSG_COL}' not found in {args.input}.")

    feature_cols = get_feature_cols(cols)
    log(f"  Feature columns sent to LLM ({len(feature_cols)}): {feature_cols}")
    log(f"  Provider: {args.provider}")

    total = len(rows)
    processed = []
    completed = 0

    log(f"Running model={args.model} prompt={script_name} on {total} rows with {args.workers} worker(s) …\n")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_row, row, args.model, system_prompt, build_prompt_fn,
                             feature_cols, num_predict, parse_fn, seed, response_format,
                             repeat_penalty, args.provider): i
            for i, row in enumerate(rows)
        }
        for future in as_completed(futures):
            result = future.result()
            processed.append(result)
            completed += 1
            if completed % 100 == 0 or completed == total:
                n_dis = sum(1 for r in processed if r.get("llm_divergence") == "divergent")
                log(f"  {completed}/{total} done  |  divergent so far: {n_dis}")

    llm_keys = []
    for r in processed:
        for k in r:
            if k.startswith("llm_") and k not in llm_keys:
                llm_keys.append(k)

    out_cols = list(cols) + llm_keys
    model_slug = args.model.replace(":", "-").replace("/", "-")
    out_file = args.output or f"results_{script_name}_{model_slug}.xlsx"

    log(f"\nWriting {out_file} …")
    wb_out = openpyxl.Workbook()
    ws_out = wb_out.active
    ws_out.title = "Results"
    ws_out.append(out_cols)
    for row in processed:
        ws_out.append([str(row.get(col, "")) for col in out_cols])
    wb_out.save(out_file)

    n_divergent = sum(1 for r in processed if r.get("llm_divergence") == "divergent")
    n_error     = sum(1 for r in processed if r.get("llm_divergence") == "error")
    n_skipped   = sum(1 for r in processed if r.get("llm_reason") == "No review messages or vote signals")
    n_nondiv    = total - n_divergent - n_skipped - n_error

    category_counts = {}
    for r in processed:
        c = r.get("llm_type_category", "non-divergent")
        category_counts[c] = category_counts.get(c, 0) + 1

    print()
    print("=" * 55)
    print(f"  Model               : {args.model}  ({args.provider})")
    print(f"  Prompt              : {script_name}")
    print(f"  Input rows          : {total}")
    print(f"  Divergent           : {n_divergent}")
    print(f"  Non-divergent       : {n_nondiv}")
    print(f"  Skipped (no signal) : {n_skipped}")
    print(f"  Errors (call/parse failed, excluded from scoring) : {n_error}")
    print()
    print("  Normalized type_category (used for scoring; raw llm_type preserved separately):")
    base_types = ["technical", "procedural", "coordination", "social", "non-divergent", "error"]
    for t in base_types:
        print(f"    {t:<14} : {category_counts.get(t, 0)}")
    dual_types = {t: c for t, c in category_counts.items() if t not in base_types}
    if dual_types:
        print("  Dual labels (e.g. technical/procedural):")
        for t, c in dual_types.items():
            print(f"    {t:<20} : {c}")
    print()
    print(f"  Output written      : {out_file}")
    print("=" * 55)