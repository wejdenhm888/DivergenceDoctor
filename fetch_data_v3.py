
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests")

try:
    import openpyxl
except ImportError:
    sys.exit("Install openpyxl first:  pip install openpyxl")


# ─────────────────────────── CONFIG ───────────────────────────
GERRIT_BASE  = "https://review.opendev.org"
MAX_CHANGES  = 1500
PAGE_SIZE    = 100
RETRY_LIMIT  = 3
RETRY_DELAY  = 5
AFTER_DATE   = "2020-01-01"
PROJECT = "openstack/cinder"

# Labels that count as human review votes (divergence signal)
SCORE_LABELS = ["Code-Review"]

# Labels produced by CI/CD infra (captured as signals, not divergence)
CI_LABELS = ["Verified", "Workflow", "Gate", "Check", "Build"]

AI_ACTORS = [
    "[bot]", "bot", "github-actions", "ci", "dependabot",
    "coderabbitai", "copilot", "claassistant", "drahtbot", "drahbot",
    "acts-project-service", "codecov-commenter", "coveralls",
    "istio-policy-bot", "lightdash-bot", "metamaskbot", "modularbot",
    "nf-core-bot", "pulumi-bot", "pytorchmergebot", "raycastbot",
    "scrutinizer-notifier", "slangbot", "strapi-cla", "wolfssl-bot",
    "owidbot", "zwave-js-bot", "pyvista-bot", "robobun", "autogpt-agent",
    "parseplatformorg", "istio-testing", "msftgits", "platform-cane891",
    "vs-mobiletools-engineering-service2", "opgitgovernance",
    "zuul", "jenkins", "automation", "openstack-gerrit", "infra-root",
    "release", "propose", "sync", "recheck", "cinder-jenkins",
    "nova-jenkins", "neutron-jenkins", "check", "gate",
    "zuul-ci", "zuul-bot", "zuul[bot]", "zuul-ci[bot]", "zuul-bot[bot]",
    "sanitybot", "qtci", "qt-ci", "qt_sanity_bot",
    "qtqa", "qt-submodule-update-bot", "qtrepotools",
    "Qt Sanity Bot", "Qt Doc Bot", "Qt Continuous Integration System",
    "openstackci", "openstack-infra", "openstackinfra",
    "gerritbot", "reviewer", "proposal-bot", "requirements",
    "requirements-bot", "stable-maint", "translation-bot",
    "openstack-release", "release-bot", "reno", "nodepool",
    "elastic-recheck", "project-config", "openstack-health",
    "airship-zuul", "starlingx-zuul",
]
_AI_ACTORS_SET = {a.lower() for a in AI_ACTORS}


# ─────────────────────────── REGEXES ──────────────────────────

# Matches the vote header line: "Patch Set N: Code-Review+1"
_VOTE_HEADER_RE     = re.compile(r"^Patch Set \d+:([ \t]+([\w\-]+[+-]\d+))*\s*\n?", re.IGNORECASE)
_NOISE_BLOCK_RE     = re.compile(r"^\s*(Outdated Votes|Copied Votes|Patch Set \d+:.*Votes.*):?\s*$", re.IGNORECASE)
_VOTE_BULLET_RE     = re.compile(r"^\s*\*\s*[\w\-]+[+-]\d+", re.IGNORECASE)
_COMMENT_COUNT_RE   = re.compile(r"^\(\d+ comments?\)\s*$", re.IGNORECASE)
_IS_VOTE_MESSAGE_RE = re.compile(r"^Patch Set \d+:\s+[\w\-]+[+-]\d+", re.IGNORECASE)
_LABEL_SCORE_RE     = re.compile(r"([\w\-]+)([+-]\d+)", re.IGNORECASE)
_PS_NUM_RE          = re.compile(r"^Patch Set (\d+):", re.IGNORECASE)

# Pure upload notification — author's own automated message, not a review
_UPLOAD_RE = re.compile(r"^Uploaded patch set \d+\.?\s*$", re.IGNORECASE)

# Commit message metadata
_DEPENDS_ON_RE = re.compile(r"Depends-On:\s*(I[0-9a-f]{40}|https?://[^\s]+)", re.IGNORECASE)
_NEEDED_BY_RE  = re.compile(r"Needed-By:\s*(I[0-9a-f]{40}|https?://[^\s]+)", re.IGNORECASE)
_CI_JOB_RE     = re.compile(r"(https?://\S+)\s+:\s+(\w+)\s+in\s+[\dhms]+", re.IGNORECASE)
_CLOSES_BUG_RE = re.compile(r"(?:closes[-\s]bug|fixed[-\s]bug|bug)[:\s#]+\d{5,10}", re.IGNORECASE)
_BLUEPRINT_RE  = re.compile(r"implements[:\s]+blueprint[:\s]+([\w\-]+)", re.IGNORECASE)
_MILESTONE_RE  = re.compile(r"milestone[:\s]+([\w\.\-]+)", re.IGNORECASE)

# ─────────────────────────── HELPERS ──────────────────────────

def normalize_timestamp(ts: str) -> str:
    if not ts:
        return ""
    ts = ts.strip().rstrip("Z").replace("T", " ").replace("Z", "")
    ts = re.sub(r"\.(\d{6})\d+", r".\1", ts)
    for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(ts, fmt).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            continue
    m = re.search(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", ts)
    if m:
        try:
            return datetime.strptime(
                m.group(1).replace("T", " "), "%Y-%m-%d %H:%M:%S"
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    return ""


def get_actor_type(actor: dict) -> str:
    name  = (actor.get("name")     or "").lower()
    uname = (actor.get("username") or "").lower()

    if not (name or uname):
        return "bot"

    def matches_keyword(text, kw):
       
        return bool(re.search(rf'(?:^|[\-_.\s]){re.escape(kw)}(?:$|[\-_.\s])', text))

    for kw in _AI_ACTORS_SET:
        if matches_keyword(name, kw) or matches_keyword(uname, kw):
            return "bot"

    if uname.endswith("[bot]") or name.endswith("[bot]"):
        return "bot"
    if uname.endswith("-ci") or name.endswith("-ci"):
        return "bot"
    if uname.endswith("-jenkins") or name.endswith("-jenkins"):
        return "bot"

    return "human"


def _clean_message_body(message_text: str) -> str:
    """Strip the vote header and noise blocks; return only genuine prose."""
    text  = _VOTE_HEADER_RE.sub("", message_text).strip()
    lines = text.splitlines()
    clean = []
    in_noise = False
    for line in lines:
        if _NOISE_BLOCK_RE.match(line):
            in_noise = True
            continue
        if in_noise:
            if _VOTE_BULLET_RE.match(line) or line.strip() == "":
                continue
            in_noise = False
        if _COMMENT_COUNT_RE.match(line.strip()):
            continue
        clean.append(line)
    return "\n".join(clean).strip()


def _is_noise_prose(prose: str) -> bool:
    noise_patterns = [
        # "Uploaded patch set 6."
        re.compile(r"^Uploaded patch set \d+\.\s*$", re.IGNORECASE),
        # "Uploaded patch set 6: Patch Set 5 was rebased."
        re.compile(r"^Uploaded patch set \d+:\s*Patch Set \d+ was rebased\.\s*$", re.IGNORECASE),
        # "Uploaded patch set 16: New patch set was added with same tree..."
        re.compile(r"^Uploaded patch set \d+:\s*New patch set was added with same tree", re.IGNORECASE),
        # "Uploaded patch set N: Commit message was updated."
        re.compile(r"^Uploaded patch set \d+:\s*Commit message was updated\.\s*$", re.IGNORECASE),
        # "Uploaded patch set N: <anything>" — catch-all for remaining upload variants
        re.compile(r"^Uploaded patch set \d+:", re.IGNORECASE),
        # other noise
        re.compile(r"^Patch Set \d+ was not helpful\.\s*$", re.IGNORECASE),
        re.compile(r"^recheck\b", re.IGNORECASE),
        re.compile(r"^reverified\b", re.IGNORECASE),
        re.compile(r"^Build (succeeded|failed|started)", re.IGNORECASE),
    ]
    for pat in noise_patterns:
        if pat.match(prose.strip()):
            return True
    return False
# ─────────────────────────── EXTRACTION ───────────────────────

def extract_all_from_messages(change: dict, owner_username: str):
    """
    Parse all Gerrit messages for a change.

    Returns:
        vote_history      : {reviewer -> {label -> [(ps_num, score, ts)]}}
        reviewer_messages : {reviewer -> [{"text", "date", "ps_num"}]}
        reviewer_types    : {reviewer -> "human"|"bot"}
        ci_signals        : list[str]
        dependency_chains : list[str]
        release_info      : list[str]
    """
    vote_history      = defaultdict(lambda: defaultdict(list))
    reviewer_messages = defaultdict(list)
    reviewer_types    = {}
    ci_signals        = []
    dependency_chains = []
    release_info_set  = set()

    # ── Commit message metadata (Depends-On, release tags) ──
    revisions = change.get("revisions", {})
    commit_msg = ""
    if revisions:
        for rev in sorted(revisions.values(), key=lambda r: r.get("_number", 0), reverse=True):
            cm = rev.get("commit", {}).get("message", "")
            if cm:
                commit_msg = cm
                break

    for m in _DEPENDS_ON_RE.finditer(commit_msg):
        dependency_chains.append(f"Depends-On:{m.group(1)}")
    for m in _NEEDED_BY_RE.finditer(commit_msg):
        dependency_chains.append(f"Needed-By:{m.group(1)}")
    if _CLOSES_BUG_RE.search(commit_msg):
        release_info_set.add("bug-fix")
    for m in _BLUEPRINT_RE.finditer(commit_msg):
        release_info_set.add(f"blueprint:{m.group(1)}")
    for m in _MILESTONE_RE.finditer(commit_msg):
        release_info_set.add(f"milestone:{m.group(1)}")

    # ── Parse each message ──
    for msg in change.get("messages", []):
        author = msg.get("author", {})
        if not author:
            continue

        text = (msg.get("message") or "").strip()
        if not text:
            continue

        date_norm = normalize_timestamp(msg.get("date", "") or "")
        atype     = get_actor_type(author)
        reviewer  = author.get("username") or author.get("name") or "unknown"

        ps_match = _PS_NUM_RE.match(text)
        ps_num   = int(ps_match.group(1)) if ps_match else None

        # ── CI signals (from bots and humans alike) ──
        header_line = text.splitlines()[0]
        ci_votes = [
            f"{lbl}{sc}"
            for lbl, sc in _LABEL_SCORE_RE.findall(header_line)
            if lbl in CI_LABELS
        ]
        if ci_votes:
            ci_signals.append(f"[{reviewer}|PS{ps_num}|{date_norm}] " + " ".join(ci_votes))
        for jm in _CI_JOB_RE.finditer(text):
            ci_signals.append(f"[{reviewer}|PS{ps_num}|{date_norm}] job:{jm.group(1)}={jm.group(2)}")

        # ── Skip bots for human review tracking ──
        if atype == "bot":
            continue

        # ── Change owner: capture prose for all_messages but skip vote tracking ──
        if reviewer == owner_username:
            prose = _clean_message_body(text)
            if prose and not _is_noise_prose(prose):
                # Tag owner messages distinctly so LLM can identify author↔reviewer conflict
                reviewer_messages[f"__owner__{reviewer}"].append({
                    "text":   prose,
                    "date":   date_norm,
                    "ps_num": ps_num,
                })
            continue

        reviewer_types[reviewer] = atype

        # ── Human vote lines ──
        if _IS_VOTE_MESSAGE_RE.match(text):
            for label, score_str in _LABEL_SCORE_RE.findall(header_line):
                if label in SCORE_LABELS:
                    vote_history[reviewer][label].append(
                        (ps_num, int(score_str), date_norm)
                    )

        # ── Prose body — skip noise ──
        prose = _clean_message_body(text)
        if prose and not _is_noise_prose(prose):
            reviewer_messages[reviewer].append({
                "text":   prose,
                "date":   date_norm,
                "ps_num": ps_num,
            })

    return (
        vote_history, reviewer_messages, reviewer_types,
        ci_signals, dependency_chains, list(release_info_set),
    )


# ─────────────────────────── DIVERGENCE ───────────────────────

def analyze_disagreements(vote_history: dict, reviewer_types: dict) -> dict:
    """
    Detect divergence across patchsets and build vote timeline.

    divergence_at_patchset : first PS number where conflicting signs coexist
                             SIMULTANEOUSLY between different reviewers.
    vote_timeline          : one entry per reviewer × PS × label with timestamp
    """
    voter_parts    = []
    diverge_at_ps  = []
    timeline_parts = []

    for label in SCORE_LABELS:
        per_reviewer = {
            r: data[label]
            for r, data in vote_history.items()
            if label in data and reviewer_types.get(r) != "bot"
        }
        if not per_reviewer:
            continue

        for reviewer, entries in sorted(per_reviewer.items()):
            atype = reviewer_types.get(reviewer, "unknown")
            for ps_num, score, ts in entries:
                timeline_parts.append(
                    f"[{reviewer}|{atype}|{label}|PS{ps_num}|{ts}]{score:+d}"
                )
            scores_str = ",".join(f"PS{p}:{s:+d}" for p, s, _ in entries)
            voter_parts.append(f"{reviewer}[{atype}]({scores_str})")

        # Walk patchsets in order
        # For each PS, take each reviewer's LATEST vote up to that PS
        # then check if signs conflict simultaneously
        all_ps = sorted({
            ps for entries in per_reviewer.values()
            for ps, _, _ in entries if ps is not None
        })

        for ps in all_ps:
            # Each reviewer's most recent score AT or BEFORE this PS
            active_scores = {}
            for reviewer, entries in per_reviewer.items():
                past = [(ep, es) for ep, es, _ in entries if ep <= ps]
                if past:
                    # take the latest vote up to this PS
                    latest_score = max(past, key=lambda x: x[0])[1]
                    active_scores[reviewer] = latest_score

            if len(active_scores) < 2:
                continue

            signs = {"+" if s > 0 else "-" for s in active_scores.values()}
            if len(signs) > 1 and ps not in diverge_at_ps:
                diverge_at_ps.append(ps)

        num_reviewers = len(per_reviewer)
        divergent     = len(diverge_at_ps) > 0

        if divergent:
            agreement = "disagree"
        elif num_reviewers == 1:
            agreement = "single"
        else:
            agreement = "agree"

        return {
            "agreement":              agreement,
            "divergent":              divergent,
            "voters_summary":         "; ".join(voter_parts),
            "vote_timeline":          " | ".join(timeline_parts),
            "divergence_at_patchset": ",".join(str(p) for p in sorted(diverge_at_ps)),
        }

    return {
        "agreement": "none", "divergent": False,
        "voters_summary": "", "vote_timeline": "", "divergence_at_patchset": "",
    }

# ─────────────────────────── METADATA BUILDERS ────────────────

def build_review_scores_detail(vote_history: dict, reviewer_types: dict) -> str:
    parts = []
    for reviewer, labels in vote_history.items():
        atype = reviewer_types.get(reviewer, "unknown")
        for label, entries in labels.items():
            for ps_num, score, ts in entries:
                parts.append(f"{label}:{reviewer}[{atype}]|PS{ps_num}:{score:+d}@{ts}")
    return " ||| ".join(parts)


def build_patch_metadata(change: dict) -> str:
    revisions      = change.get("revisions", {})
    patchset_count = len(revisions)
    insertions     = change.get("insertions", "?")
    deletions      = change.get("deletions",  "?")
    file_count     = "?"
    if revisions:
        latest = max(revisions.values(), key=lambda r: r.get("_number", 0))
        files  = latest.get("files", {})
        if files:
            file_count = len(files)
    return (f"patchsets:{patchset_count} | files:{file_count} "
            f"| insertions:{insertions} | deletions:{deletions}")


def build_related_patches(change: dict) -> str:
    parts = []
    topic  = change.get("topic", "")
    sub_id = change.get("submission_id", "")
    if topic:
        parts.append(f"topic:{topic}")
    if sub_id:
        parts.append(f"submission_id:{sub_id}")
    return " | ".join(parts)


# ─────────────────────────── HTTP ─────────────────────────────

def _gerrit_get(url: str, params=None):
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            text = r.text
            if text.startswith(")]}'"):
                text = text[4:]
            return json.loads(text)
        except Exception as exc:
            print(f"  [warn] {url} attempt {attempt}/{RETRY_LIMIT} failed: {exc}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
    return None


def fetch_all_changes(target: int, status: str = "merged",
                      after: str = AFTER_DATE) -> list:
    all_results, start = [], 0
    print(f"  Target : {target} | Server : {GERRIT_BASE} | After : {after}\n")

    while len(all_results) < target:
        want   = min(PAGE_SIZE, target - len(all_results))
        params = {
            "q": f"status:{status} project:{PROJECT} after:{after}",     
           "n": want,
            "S": start,
            "o": [
                "DETAILED_LABELS", "MESSAGES", "DETAILED_ACCOUNTS",
                "CURRENT_REVISION", "ALL_REVISIONS",
                "CURRENT_COMMIT",   "ALL_COMMITS",
                "CURRENT_FILES",    "SUBMITTABLE", "TRACKING_IDS",
            ],
        }
        chunk = None
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                r = requests.get(f"{GERRIT_BASE}/changes/", params=params, timeout=60)
                r.raise_for_status()
                text = r.text
                if text.startswith(")]}'"):
                    text = text[4:]
                chunk = json.loads(text)
                break
            except Exception as exc:
                print(f"  [warn] S={start} attempt {attempt}/{RETRY_LIMIT} failed: {exc}")
                if attempt < RETRY_LIMIT:
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  [error] giving up at S={start}.")
                    return all_results

        if not chunk:
            print("  Empty page — stopping.")
            break

        all_results.extend(chunk)
        start   += len(chunk)
        has_more = chunk[-1].get("_more_changes", False)
        print(f"  [{time.strftime('%H:%M:%S')}] Fetched {len(all_results)}/{target}"
              f"  (+{len(chunk)} | more={has_more})")
        if not has_more:
            print("  Server reports no more changes.")
            break
        time.sleep(0.3)

    return all_results[:target]


def fetch_inline_comments(change_num: int, revision_id: str) -> dict:
    url  = f"{GERRIT_BASE}/changes/{change_num}/revisions/{revision_id}/comments"
    data = _gerrit_get(url)
    return data if isinstance(data, dict) else {}



def main():
    print("=== Gerrit Review Fetcher v3 ===\n")
    changes = fetch_all_changes(MAX_CHANGES, status="merged", after=AFTER_DATE)
    total   = len(changes)
    print(f"\nFetch complete: {total} changes. Processing …\n")

    rows = []

    for idx, ch in enumerate(changes, start=1):
        change_num = ch.get("_number", "")
        subject    = ch.get("subject", "")
        project    = ch.get("project", "")
        branch     = ch.get("branch", "")
        owner      = ch.get("owner", {})
        owner_name = owner.get("username") or owner.get("name") or ""
        owner_type = get_actor_type(owner)
        updated    = ch.get("updated", "")
        url        = f"{GERRIT_BASE}/c/{project}/+/{change_num}"

        # ── Messages, votes, CI signals, metadata ──
        (vote_history, reviewer_messages, reviewer_types,
         ci_signals, dependency_chains, release_info) = \
            extract_all_from_messages(ch, owner_name)

        analysis       = analyze_disagreements(vote_history, reviewer_types)
        patch_meta     = build_patch_metadata(ch)
        patchset_count = len(ch.get("revisions", {}))
        related        = build_related_patches(ch)

        # Tracking IDs from Gerrit (Launchpad bugs, StoryBoard stories)
        for tid in ch.get("tracking_ids", []):
            system = (tid.get("system", "") or "").lower()
            if system:
                # tracking_ids confirm the type, don't repeat the raw ID
                if "launchpad" in system or "bug" in system:
                    if "bug-fix" not in release_info:
                        release_info.append("bug-fix")
                elif "storyboard" in system:
                    if "storyboard-task" not in release_info:
                        release_info.append("storyboard-task")

        review_scores_detail = build_review_scores_detail(vote_history, reviewer_types)

        # ── Inline comments (per patchset, bots and owner excluded) ──
        inline_parts = []
        revisions    = ch.get("revisions", {})
        for rev_sha, rev_info in sorted(revisions.items(),
                                        key=lambda kv: kv[1].get("_number", 0)):
            rev_number  = rev_info.get("_number", "?")
            rev_created = rev_info.get("created", "") or ""
            comments    = fetch_inline_comments(change_num, rev_sha)

            for filepath, comment_list in comments.items():
                for c in comment_list:
                    author   = c.get("author", {})
                    reviewer = author.get("username") or author.get("name") or "unknown"
                    atype    = get_actor_type(author)

                    # Skip bots
                    if atype == "bot":
                        continue

                    # Tag owner inline comments distinctly
                    if reviewer == owner_name:
                        atype = "owner"

                    line     = c.get("line", "?")
                    msg      = (c.get("message") or "").strip()
                    ts       = normalize_timestamp(
                        c.get("updated") or c.get("created") or rev_created or ""
                    )
                    in_reply = bool(c.get("in_reply_to"))
                    tag      = "inline-reply" if in_reply else "inline"
                    if msg and not _is_noise_prose(msg):
                        inline_parts.append(
                            f"[{tag}|{reviewer}|{atype}|PS{rev_number}"
                            f"|{filepath}:{line}|{ts}] {msg}"
                        )

        # ── Top-level prose messages ──
        top_level_parts = []
        for reviewer_key, prose_list in reviewer_messages.items():
            # Owner messages are stored with __owner__ prefix
            is_owner  = reviewer_key.startswith("__owner__")
            reviewer  = reviewer_key[len("__owner__"):] if is_owner else reviewer_key
            atype     = "owner" if is_owner else reviewer_types.get(reviewer_key, "unknown")
            for entry in prose_list:
                prose  = (entry.get("text") or "").strip()
                date   = entry.get("date", "")
                ps_num = entry.get("ps_num")
                if not prose:
                    continue
                ps_tag = f"|PS{ps_num}" if ps_num is not None else ""
                top_level_parts.append(
                    f"[top-level|{reviewer}|{atype}{ps_tag}|{date}] {prose}"
                )

        # ── Actor flags ──
        types_present   = sorted(set(reviewer_types.values()))
        has_bot_voter   = "bot"   in reviewer_types.values()
        has_human_voter = "human" in reviewer_types.values()

        # ── all_messages = top-level + inline (input to LLM script) ──
        all_messages = " ||| ".join(top_level_parts + inline_parts)

        rows.append({
            "change_number":          change_num,
            "subject":                subject,
            "project":                project,
            "branch":                 branch,
            "owner":                  owner_name,
            "owner_type":             owner_type,
            "updated":                updated,
            "url":                    url,
            "label":                  SCORE_LABELS[0],
            # vote summary
            "all_votes":              analysis["voters_summary"],
            "agreement":              analysis["agreement"],
            "divergent":              analysis["divergent"],
            # patchset-level
            "patchset_count":         patchset_count,
            "divergence_at_patchset": analysis["divergence_at_patchset"],
            # vote detail
            "vote_timeline":          analysis["vote_timeline"],
            "review_scores_detail":   review_scores_detail,
            # actor flags
            "has_bot_voter":          has_bot_voter,
            "has_human_voter":        has_human_voter,
            "actor_types":            ",".join(types_present),
            # typed messages
            "top_level_messages":     " ||| ".join(top_level_parts),
            "inline_comments":        " ||| ".join(inline_parts),
            "all_messages":           all_messages,
            # rich metadata
            "patch_metadata":         patch_meta,
            "dependency_chains":      " | ".join(dependency_chains),
            "related_patches":        related,
            "release_info":           " | ".join(release_info),
            "ci_cd_signals":          " ||| ".join(ci_signals),
        })

        if idx % 100 == 0 or idx == total:
            print(f"[{time.strftime('%H:%M:%S')}] Processed {idx}/{total}")

    fieldnames = [
        "change_number", "subject", "project", "branch",
        "owner", "owner_type", "updated", "url", "label",
        "all_votes", "agreement", "divergent",
        "patchset_count", "divergence_at_patchset",
        "vote_timeline", "review_scores_detail",
        "has_bot_voter", "has_human_voter", "actor_types",
        "top_level_messages", "inline_comments", "all_messages",
        "patch_metadata", "dependency_chains", "related_patches",
        "release_info", "ci_cd_signals",
    ]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Reviews"
    ws.append(fieldnames)
    for row in rows:
        ws.append([row.get(col, "") for col in fieldnames])
    out_file = "gerrit_reviews_large_cinder.xlsx"
    wb.save(out_file)
    agree       = sum(1 for r in rows if r["agreement"] == "agree")
    disagree    = sum(1 for r in rows if r["agreement"] == "disagree")
    single      = sum(1 for r in rows if r["agreement"] == "single")
    none_       = sum(1 for r in rows if r["agreement"] == "none")
    divergent   = sum(1 for r in rows if r["divergent"])
    with_top    = sum(1 for r in rows if r["top_level_messages"])
    with_inline = sum(1 for r in rows if r["inline_comments"])
    with_deps   = sum(1 for r in rows if r["dependency_chains"])
    with_ci     = sum(1 for r in rows if r["ci_cd_signals"])

    print(f"\nDone. Wrote {len(rows)} rows → '{out_file}'")
    print("\nAgreement breakdown:")
    print(f"  agree     : {agree}")
    print(f"  disagree  : {disagree}")
    print(f"  single    : {single}")
    print(f"  none      : {none_}")
    print(f"  divergent : {divergent}")
    print("\nMessage coverage (after noise filtering):")
    print(f"  changes with top-level reviewer prose : {with_top}")
    print(f"  changes with inline comments          : {with_inline}")
    print(f"  changes with Depends-On               : {with_deps}")
    print(f"  changes with CI/CD signals            : {with_ci}")
    print("\nColumn guide:")
    print("  top_level_messages     — [top-level|reviewer|human|PS<n>|ts] genuine prose only")
    print("  inline_comments        — [inline|reviewer|human|PS<n>|file:line|ts] prose")
    print("  all_messages           — top-level + inline merged (input to LLM script)")
    print("  patchset_count         — total patchsets uploaded")
    print("  divergence_at_patchset — PS number where conflicting votes first appeared")
    print("  vote_timeline          — [reviewer|type|label|PS<n>|ts]<score> per vote event")
    print("  review_scores_detail   — label:reviewer[type]|PS<n>:<score>@<ts> per vote")
    print("  patch_metadata         — patchsets / files / insertions / deletions")
    print("  dependency_chains      — Depends-On / Needed-By from commit message")
    print("  related_patches        — topic / submission_id")
    print("  release_info           — milestone / bug IDs / blueprint references")
    print("  ci_cd_signals          — Verified/Workflow votes + Zuul job results")


if __name__ == "__main__":
    main()