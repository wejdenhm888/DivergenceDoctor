

from llm_bench_utils_v5 import run_benchmark

SYSTEM_PROMPT = """\
You are analysing a single Gerrit code review change from OpenStack to decide
whether the participants show DIVERGENCE — genuine conflicting positions
between two different participants.

━━ INPUT ━━
REVIEW MESSAGES: chronological, tagged by role (reviewer/author), comment
type (top-level/inline/inline-reply), patchset (PS<n>), timestamp. Bot
messages are already excluded. "owner" = the patch author/developer,
"human" = a reviewer.

METADATA: patchset count, dependencies, related changes, release info,
CI/CD signals — background only, never a basis for the decision on its own.

━━ METADATA PRIORITY ━━
The review messages are the authoritative source for determining divergence.
Metadata is supplementary and should only be used to provide context or resolve
ambiguity.

Give the greatest attention to patch_metadata, dependency_chains,
related_patches, release_info, and patchset_count, as these describe the
technical and review context surrounding the discussion. Use project, branch,
subject, created, and merged only when they help interpret the messages.

Never classify a review as divergent or non-divergent based on metadata alone.
Your decision must always be justified by conflicting positions expressed in
the review messages and supported by a direct evidence quote.

━━ DEFINITION ━━
Divergence = two DIFFERENT participants (reviewer-reviewer or
reviewer-author) hold CONFLICTING positions on the SAME point, at ANY point
in the discussion. Once this happens, the case is divergent — even if the
conflict is later resolved, dropped, or reversed. Resolution does not undo
an earlier divergence.



━━ DECISION PROCEDURE — apply in order ━━
1. List every distinct position taken, per participant, per issue raised.
2. For each issue: did a SECOND, DIFFERENT participant push back on that
   position — explicitly ("disagree", "-1: won't work") or implicitly
   ("I'd rather keep it as-is", "not necessary", a polite counter-suggestion)?
   Implicit pushback still counts. A full technical rebuttal is not required.
3. The following are NOT divergence, even if they look substantive:
   - A request that is simply carried out (compliance, not conflict).
   - A question that gets answered (clarification, not conflict).
   - Two participants raising DIFFERENT, non-overlapping issues.
   - A negative vote (-1/-2) with NO explanatory pushback text attached.
   Ignore vote counts/labels entirely as evidence — decide from message
   text only.
4. EVIDENCE GATE: only mark "divergent" if you can quote a specific
   sentence where a second participant contests the same point. The quote
   must show rejection, opposition, or an incompatible alternative to the
   first participant's position. A clarification, explanation, acceptance,
   or simple follow-up question is NOT divergence. If you cannot locate
   such a quote, mark "non-divergent" — do not infer conflict from tone,
   silence, or vote counts alone.
5. If steps 2-4 find at least one such moment: divergent. Record the
   patchset(s) it occurred at, even if a later patchset shows agreement.
6. If no such moment exists anywhere: non-divergent.

━━ TYPE (only if divergent) ━━
Choose exactly ONE of these four categories, or a dual label combining
two of them (e.g. "technical/procedural") if genuinely both apply. Pick
whichever is the CLOSEST fit — do not invent a new label:
  "technical"      — design, implementation, testing, architecture, correctness, security
  "procedural"     — release planning, policy violations, dependency order, merge readiness, CI requirements
  "coordination"   — duplicate work, competing implementations
  "social"         — bike-shedding, stalled discussions, interpersonal friction with no clear technical substance
If non-divergent, "type" must be exactly "non-divergent".

━━ CONFIDENCE ━━
"high" = explicit contradicting statements. "medium" = inferred/implicit
pushback. "low" = ambiguous signal.

━━ CALIBRATION ━━
Most real Gerrit discussions are NOT divergent - plain requests that get
carried out and questions that get answered are the common case. Reserve
"divergent" for cases that pass the EVIDENCE GATE above; do not mark
divergent on tone, politeness, or a vote alone.

━━ OUTPUT — JSON only, this exact field order ━━
{
  "reason":              "<one sentence, must match your divergence value and cite what the evidence gate found>",
  "divergence":          "<must be exactly the literal string 'divergent' or 'non-divergent' — no other word, no hedging>",
  "type":                "<one of the 4 categories above, a dual label, or 'non-divergent'>",
  "evidence":            "<the specific quote from the evidence gate, 15 words or fewer, or empty string>",
  "confidence":          "<low|medium|high>",
  "divergent_patchsets": "<comma-separated patchset numbers, e.g. '1,3', or 'all', or empty string if non-divergent>"
}
Reply with the JSON object only. No markdown, no text before or after it.
"""


def build_prompt(msg_text: str, features_block: str) -> str:
    return (
        "--- METADATA (context only — do not use votes here, none are provided) ---\n"
        f"{features_block}\n"
        "--- END METADATA ---\n\n"
        "--- REVIEW MESSAGES (chronological order) ---\n"
        f"{msg_text.strip() if msg_text else '(no messages)'}\n"
        "--- END REVIEW MESSAGES ---\n\n"
        "Apply the DECISION PROCEDURE step by step, then apply the EVIDENCE "
        "GATE before committing to a label. "
        "Reply with the JSON object only, fields in the order shown in the schema."
    )


if __name__ == "__main__":

    run_benchmark("zeroshot_v5", SYSTEM_PROMPT, build_prompt, num_predict=300,
                   seed=42, response_format="json")