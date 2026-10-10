"""
fetch_single_change.py
"""

import time
import json
import argparse
import requests

from fetch_data_v3 import (
    GERRIT_BASE, extract_all_from_messages, analyze_disagreements,
    build_patch_metadata, build_related_patches, build_review_scores_detail,
    fetch_inline_comments, get_actor_type, normalize_timestamp, _is_noise_prose,
)


def fetch_single_change_row(change_number) -> dict:
    """Mirror of the per-change body of fetch-data.py's main(), for one change_id."""
    url = f"{GERRIT_BASE}/changes/{change_number}/"
    params = {
        "o": [
            "DETAILED_LABELS", "MESSAGES", "DETAILED_ACCOUNTS",
            "CURRENT_REVISION", "ALL_REVISIONS",
            "CURRENT_COMMIT", "ALL_COMMITS",
            "CURRENT_FILES", "SUBMITTABLE", "TRACKING_IDS",
        ]
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    text = r.text
    if text.startswith(")]}'"):
        text = text[4:]
    ch = json.loads(text)

    change_num = ch.get("_number", change_number)
    subject = ch.get("subject", "")
    project = ch.get("project", "")
    branch = ch.get("branch", "")
    owner = ch.get("owner", {})
    owner_name = owner.get("username") or owner.get("name") or ""
    owner_type = get_actor_type(owner)
    updated = ch.get("updated", "")
    change_url = f"{GERRIT_BASE}/c/{project}/+/{change_num}"

    (vote_history, reviewer_messages, reviewer_types,
     ci_signals, dependency_chains, release_info) = \
        extract_all_from_messages(ch, owner_name)

    analysis = analyze_disagreements(vote_history, reviewer_types)
    patch_meta = build_patch_metadata(ch)
    patchset_count = len(ch.get("revisions", {}))
    related = build_related_patches(ch)

    for tid in ch.get("tracking_ids", []):
        system = (tid.get("system", "") or "").lower()
        if system:
            if "launchpad" in system or "bug" in system:
                if "bug-fix" not in release_info:
                    release_info.append("bug-fix")
            elif "storyboard" in system:
                if "storyboard-task" not in release_info:
                    release_info.append("storyboard-task")

    review_scores_detail = build_review_scores_detail(vote_history, reviewer_types)

    # ── Inline comments (per patchset, bots and owner excluded from filtering out) ──
    inline_parts = []
    revisions = ch.get("revisions", {})
    for rev_sha, rev_info in sorted(revisions.items(),
                                     key=lambda kv: kv[1].get("_number", 0)):
        rev_number = rev_info.get("_number", "?")
        rev_created = rev_info.get("created", "") or ""
        print(f"  fetching inline comments for patchset {rev_number}...", flush=True)

        comments = fetch_inline_comments(change_num, rev_sha)

        for filepath, comment_list in comments.items():
            for c in comment_list:
                author = c.get("author", {})
                reviewer = author.get("username") or author.get("name") or "unknown"
                atype = get_actor_type(author)
                if atype == "bot":
                    continue
                if reviewer == owner_name:
                    atype = "owner"
                line = c.get("line", "?")
                msg = (c.get("message") or "").strip()
                ts = normalize_timestamp(c.get("updated") or c.get("created") or rev_created or "")
                in_reply = bool(c.get("in_reply_to"))
                tag = "inline-reply" if in_reply else "inline"
                if msg and not _is_noise_prose(msg):
                    inline_parts.append(
                        f"[{tag}|{reviewer}|{atype}|PS{rev_number}|{filepath}:{line}|{ts}] {msg}"
                    )

    # ── Top-level prose messages ──
    top_level_parts = []
    for reviewer_key, prose_list in reviewer_messages.items():
        is_owner = reviewer_key.startswith("__owner__")
        reviewer = reviewer_key[len("__owner__"):] if is_owner else reviewer_key
        atype = "owner" if is_owner else reviewer_types.get(reviewer_key, "unknown")
        for entry in prose_list:
            prose = (entry.get("text") or "").strip()
            date = entry.get("date", "")
            ps_num = entry.get("ps_num")
            if not prose:
                continue
            ps_tag = f"|PS{ps_num}" if ps_num is not None else ""
            top_level_parts.append(f"[top-level|{reviewer}|{atype}{ps_tag}|{date}] {prose}")

    types_present = sorted(set(reviewer_types.values()))
    has_bot_voter = "bot" in reviewer_types.values()
    has_human_voter = "human" in reviewer_types.values()
    all_messages = " ||| ".join(top_level_parts + inline_parts)

    row = {
        "change_number": change_num,
        "subject": subject,
        "project": project,
        "branch": branch,
        "owner": owner_name,
        "owner_type": owner_type,
        "updated": updated,
        "url": change_url,
        "label": "Code-Review",
        "all_votes": analysis["voters_summary"],
        "agreement": analysis["agreement"],
        "divergent": analysis["divergent"],
        "patchset_count": str(patchset_count),
        "divergence_at_patchset": analysis["divergence_at_patchset"],
        "vote_timeline": analysis["vote_timeline"],
        "review_scores_detail": review_scores_detail,
        "has_bot_voter": str(has_bot_voter),
        "has_human_voter": str(has_human_voter),
        "actor_types": ",".join(types_present),
        "top_level_messages": " ||| ".join(top_level_parts),
        "inline_comments": " ||| ".join(inline_parts),
        "all_messages": all_messages,
        "patch_metadata": patch_meta,
        "dependency_chains": " | ".join(dependency_chains),
        "related_patches": related,
        "release_info": " | ".join(release_info),
        "ci_cd_signals": " ||| ".join(ci_signals),
    }
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("change_number", help="Gerrit change number, e.g. 123456")
    args = parser.parse_args()

    t0 = time.time()
    row = fetch_single_change_row(args.change_number)
    print(f"\nFetched change {row['change_number']} in {time.time() - t0:.1f}s")
    print(f"  subject         : {row['subject']}")
    print(f"  project/branch  : {row['project']} / {row['branch']}")
    print(f"  patchset_count  : {row['patchset_count']}")
    print(f"  agreement       : {row['agreement']}  (vote-based, context only)")
    n_top = len(row['top_level_messages'].split(' ||| ')) if row['top_level_messages'] else 0
    n_inl = len(row['inline_comments'].split(' ||| ')) if row['inline_comments'] else 0
    print(f"  top-level msgs  : {n_top}")
    print(f"  inline comments : {n_inl}")
