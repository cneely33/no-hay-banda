#!/usr/bin/env python3
"""
Talks to the marketing llama.cpp stack only (port 8018, alias
qwen38-27b-heretic-marketing-q6k). That process is VRAM-exclusive with the
diversity champion on 8017. 

The model is instructed to reverse-engineer a mimic-ready film-review
podcast persona (voice, taste, show grammar) with quote-backed traits.
Political/conspiracy asides are recorded as recognition registers, not as
New Showbiz public copy.

Usage:
  python3 build_persona.py --check
  python3 build_persona.py --dry-run
  python3 build_persona.py --limit 2
  python3 build_persona.py
  python3 build_persona.py --show "The Poz Button" --host "Borzoi"
  python3 build_persona.py --reduce-only
  python3 build_persona.py --redo-truncated
  python3 build_persona.py --persona-from /usr/src/no-hay-banda/persona-run/reduce
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

DEFAULT_SHOW = "The Poz Button"
DEFAULT_HOST = "Borzoi"
DEFAULT_TRANSCRIPTS = Path("/usr/src/no-hay-banda/transcripts")
DEFAULT_OUT = Path("/usr/src/no-hay-banda/persona-run")
DEFAULT_BASE_URL = "http://localhost:8018/v1"
DEFAULT_MODEL = "qwen38-27b-heretic-marketing-q6k"
COMPOSE = (
    "/usr/src/ai-stack/LocalLLM/llama-cpp/"
    "qwen38-27b-heretic-marketing/docker-compose.yml"
)

# Marketing compose: --ctx-size 262144, --parallel 1 → n_ctx_seq 262144.
# llama.cpp 400s when n_prompt + n_predict > n_ctx. Do not set max_tokens to
# leftover room: the model fills it, fold files bloat, and extracts still hit
# finish_reason=length when leftover was small under 65k. Cap completions;
# chunk and pack so prompt + cap + headroom <= n_ctx.
N_CTX = 262_144
CTX_HEADROOM = 2_048
EXTRACT_MAX_TOKENS = 24_576
REDUCE_MAX_TOKENS = 8_192
PERSONA_MAX_TOKENS = 24_576
CHUNK_OVERLAP = 1_200
HTTP_TIMEOUT_S = 3_600
TRUNCATION_MARK = "[truncated: finish_reason=length]"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_REASONING_TAG = re.compile(
    r"<(?:reasoning|analysis)>.*?</(?:reasoning|analysis)>",
    re.IGNORECASE | re.DOTALL,
)


EXTRACT_SYSTEM = """\
You extract a film-review podcast PERSONA from one transcript chunk.

Show: [[SHOW]]
Primary host to profile: [[HOST]]

This is evidence collection, not a vibe paragraph and not a plot recap.

Rules:
- Quote short distinctive lines only (under 25 words). Cite the timestamp as printed
  in the transcript (e.g. [0:12:04]).
- Every candidate trait needs a quote. If you cannot quote it, do not claim it.
- Ban empty adjectives: witty, insightful, passionate, unique, authentic.
- Strip ums and false starts unless they are clearly a bit.
- Do not treat ASR errors as voice.
- Speakers are often unlabeled. Name speakers when the intro or address makes it
  obvious. Attribute lines to [[HOST]] only at high confidence; otherwise
  speaker: unknown.
- Profile [[HOST]], not guests. Note guest/co-host moves under Show grammar.
- Separate film-craft talk from industry talk, personal anecdote, and political
  or racial asides (register jumps). Quote register jumps; do not turn slurs or
  conspiracy claims into required catchphrases or writing commands.
- Trait score: 0 absent, 1 once, 2 would likely recur, 3 load-bearing in this chunk.
- If the chunk is bumper, ads, or dead air, say so and output a thin extract.
- Finish after Candidate traits. Do not pad.

Output markdown with exactly these headings:

# Extract
- file: (from the user header)
- chunk:
- speakers_observed:
- host_attribution_confidence: high | mixed | low

## Attention observations
Numbered craft notices (acting, camera, writing, sound, design, VFX, editing,
genre history, industry, identity-on-screen, autobiography, meme) with timestamp.

## Value verbs
Words that mean good or bad movie for this host.

## Argument shapes
thesis-first | scene-first | comparison | yes-but | co-host dialectic | other

## Quotes
Each bullet: quote | timestamp | why it matters | trait_score

## Lexicon
keep / steal-with-care / never (from this chunk only)

## Heat
Charity vs contempt; who they punch; boredom-without-theory.

## Humor
Type and one example timestamp if any.

## Spoilers and recs
## Comparisons
## Scoring
## Show grammar
Bumper, bits, who does plot vs thesis vs dissent.

## Register jumps
Film-craft vs industry vs personal vs political. Quote + timestamp.

## Candidate traits
name | score 0-3 | confidence high/mixed/low
"""

REDUCE_SYSTEM = """\
You merge film-review podcast extracts into one evidence file for [[SHOW]] / [[HOST]].

Rules:
- Keep traits that appear in multiple extracts or are clearly load-bearing (score 3).
- Drop one-off color unless it is a named running bit.
- Deduplicate quotes; keep the sharpest short quote plus file + timestamp.
- Count recurrence when you can (e.g. "alive: 4 extracts").
- Do not invent quotes, films, or catchphrases.
- Preserve the split: voice vs taste vs show grammar vs register jumps.
- Host vs co-host must stay distinct. Never average two speakers into one voice.
- Political/racial asides stay in Register jumps, not in Sentence rules.
- If extracts conflict, say mixed and keep both quotes.
- Output denser than the inputs but still evidence-first. No vibe summary.
- Finish when the schema is complete. Do not pad to fill the token budget.

Output markdown:

# Merged evidence: [[SHOW]] / [[HOST]]

## Speakers and show grammar
## Audience contract (evidence)
## Theory of movies (praise / punish / yardstick) with quotes
## Attention order (tallied)
## Argument shapes (tallied)
## Sentence music (with quotes)
## Lexicon keep / steal / never (with counts)
## Heat rules (with quotes)
## Humor (with quotes)
## Spoilers, recs, ethics
## Taste neighbors / comparisons
## Scoring behavior
## Register jumps (do not promote to writing SOP)
## Trait table
trait | max_score | recurrence | best quote | file | timestamp | keep_main|optional|drop
## Conflicts and gaps
"""

PERSONA_SYSTEM = """\
Write the final mimic-ready persona for [[SHOW]] host [[HOST]] from merged extracts.

This is a writing spec a later agent can execute without hearing the show.
It is NOT New Showbiz public voice. Include a brand wall.

Rules:
- Main body: trait score 2+ only (recurs or load-bearing). Trait-1 goes under
  Optional color. Missing sections: write `insufficient evidence`.
- Every bullet in Theory, Sentence rules, Moves, Lexicon, Heat, and Calibration
  keep-lines needs a short quote plus episode filename and timestamp from the
  extracts. No quote, no trait.
- Do not invent catchphrases, films, or biographical facts.
- Include 8 keep/reject calibration triples if evidence supports them; fewer is
  OK if you mark insufficient. Always reject generic AI-critic diction
  (delve, unpack, nuanced, tour de force, emotional heights, viewing experience).
- Third-person for Identity. Second-person "You are..." only in Sentence rules
  and Move library as writer instructions.
- No em dash in writer-instruction examples if you can use a period or comma.

Write markdown:

# Persona: [[SHOW]] / [[HOST]]

## Identity
One paragraph. Role on the show. Not a biography dump.

## Audience contract
Seen vs unseen, spoiler rule, expertise assumed.

## Theory of movies
Five praise bullets and five punish bullets, each with quote + source.
Then one yardstick paragraph.

## Attention order
Ranked list of what they notice first.

## Default review shape
Numbered beat sheet for a 400-word written recap of a spoken review.

## Sentence rules
Length, fragments, POV, profanity function, metaphor bank, forbidden words.

## Move library
Named moves they actually use. Each: do / don't / example modeled on a quote.

## Taste neighbors
Touchstone films/directors and the mechanism they stand for.

## Heat rules
Who may be dunked; who is protected; boredom policy.

## Humor rules
Type, density, running bits with a quota (max 1 per piece unless evidence says otherwise).

## Recognition registers
Cinephile-hunt register vs political/conspiracy jump: how to *spot* each, not how
to publish as New Showbiz.

## Do not impersonate
Other hosts, other shows, Composer literary voice, New Showbiz cold-reader unless
explicitly translating structure into Kakusu copy.

## Negative constraints
Things the corpus never does.

## Optional color
Trait-1 only.

## Calibration set
Keep/reject triples for: empty competent sequel; sacred cow they still like;
messy original they defend; great performance in a film they dislike; respect
but never rewatch; enjoy but will not defend as art.

## Brand wall
This file is an internal overlay and a recognition profile. Do not draft
@new_show_biz as [[HOST]]. Public copy stays cold-reader cinematic observation.
"""


def fill_prompt(template: str, show: str, host: str) -> str:
    """Inject show/host names into a system prompt template."""
    return template.replace("[[SHOW]]", show).replace("[[HOST]]", host)


def strip_reasoning(text: str) -> str:
    """Drop private chain-of-thought if the template override failed."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _REASONING_TAG.sub("", cleaned)
    return cleaned.strip()


def slug(name: str) -> str:
    """Filesystem-safe extract id; keep episode numbers readable."""
    cleaned = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)
    return cleaned.strip("._")[:180]


def iter_transcripts(root: Path) -> list[Path]:
    """All .txt files under root, nested, sorted by path."""
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def extract_chunk_chars(system: str) -> int:
    """Largest transcript slice that still leaves EXTRACT_MAX_TOKENS of room."""
    reserved = (
        CTX_HEADROOM
        + EXTRACT_MAX_TOKENS
        + estimate_tokens(system)
        + 400
    )
    token_room = max(N_CTX - reserved, 4_096)
    return max(20_000, (token_room * 22) // 10)


def chunk_text(text: str, size: int, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split a transcript so prompt + extract cap stay inside n_ctx_seq."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            window = text[start:end]
            nl = window.rfind("\n")
            if nl >= size // 2:
                end = start + nl + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class LlamaClient:
    """OpenAI-compatible client for llama-server on 8018."""

    def __init__(self, base_url: str, model: str, timeout_s: int = HTTP_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def health(self) -> tuple[bool, str]:
        health_url = self.base_url.replace("/v1", "") + "/health"
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            models_url = f"{self.base_url}/models"
            with urllib.request.urlopen(models_url, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            ids = [item.get("id", "") for item in payload.get("data", [])]
            if self.model not in ids:
                return False, (
                    f"health ok but model {self.model!r} not in {ids}. "
                    f"This script is for the marketing stack in {COMPOSE}."
                )
            return True, body.strip() or "ok"
        except urllib.error.URLError as exc:
            return False, (
                f"Cannot reach {health_url} ({exc}). "
                "Stop the 8017 diversity container first (VRAM exclusive), then:\n"
                f"  cd /usr/src/ai-stack/LocalLLM/llama-cpp/qwen38-27b-heretic-marketing "
                "&& docker compose up -d\n"
                "  curl -sS http://localhost:8018/health\n"
                "  curl -sS http://localhost:8018/v1/models"
            )

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.2,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:800]}") from exc
        choice = raw["choices"][0]
        finish = choice.get("finish_reason")
        content = choice.get("message", {}).get("content") or ""
        text = strip_reasoning(content)
        if "<think>" in content.lower():
            raise RuntimeError(
                "Model returned a <think> block. The Qwen3.8 chat template "
                "is not mounted; fix the compose bind-mount before continuing."
            )
        if finish == "length":
            text += f"\n\n{TRUNCATION_MARK}"
        return text


def extract_one(
    client: LlamaClient,
    path: Path,
    chunk: str,
    chunk_idx: int,
    chunk_total: int,
    system: str,
) -> tuple[str, int]:
    header = (
        f"Transcript file: {path.name}\n"
        f"Chunk {chunk_idx} of {chunk_total} ({len(chunk)} chars)\n"
        "Extract persona evidence from this chunk only.\n"
        "---\n"
    )
    user = f"/nothink\n{header}{chunk}"
    max_tokens = completion_budget(system, user, EXTRACT_MAX_TOKENS)
    return client.chat(system, user, max_tokens=max_tokens), max_tokens


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def is_complete(path: Path) -> bool:
    """Resume must not skip files that hit max_tokens."""
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return TRUNCATION_MARK not in text


def clear_derived(out_dir: Path) -> None:
    """Reduce/fold/persona are invalid once any extract is regenerated."""
    persona = out_dir / "persona.md"
    if persona.exists():
        persona.unlink()
        print(f"removed {persona}", flush=True)
    for child in sorted(out_dir.iterdir()):
        if child.is_dir() and child.name.startswith("reduce"):
            shutil.rmtree(child)
            print(f"removed {child}/", flush=True)


def extract_path(out_dir: Path, transcript: Path, chunk_idx: int) -> Path:
    return out_dir / "extracts" / f"{slug(transcript.stem)}__{chunk_idx:02d}.md"


def prune_stale_chunks(out_dir: Path, transcript: Path, kept: int) -> None:
    """Drop leftover __02.md… files after a transcript now fits in fewer chunks."""
    extract_dir = out_dir / "extracts"
    if not extract_dir.is_dir() or kept < 1:
        return
    prefix = f"{slug(transcript.stem)}__"
    for path in extract_dir.glob(f"{prefix}*.md"):
        stem = path.stem
        if "__" not in stem:
            continue
        idx_text = stem.rsplit("__", 1)[-1]
        if not idx_text.isdigit():
            continue
        if int(idx_text) > kept:
            path.unlink()
            print(f"removed stale {path.name}", flush=True)


def run_extracts(
    client: LlamaClient,
    transcripts: list[Path],
    out_dir: Path,
    resume: bool,
    system: str,
) -> list[Path]:
    written: list[Path] = []
    total = len(transcripts)
    chunk_size = extract_chunk_chars(system)
    print(f"extract chunk size: {chunk_size} chars (n_ctx={N_CTX})", flush=True)
    for file_i, path in enumerate(transcripts, start=1):
        chunks = chunk_text(load_text(path), size=chunk_size)
        if not chunks:
            print(f"[{file_i}/{total}] empty, skip {path.name}", flush=True)
            continue
        prune_stale_chunks(out_dir, path, len(chunks))
        print(
            f"[{file_i}/{total}] {path.name} ({len(chunks)} chunk(s))",
            flush=True,
        )
        for idx, chunk in enumerate(chunks, start=1):
            dest = extract_path(out_dir, path, idx)
            if resume and is_complete(dest):
                written.append(dest)
                continue
            started = time.monotonic()
            text, max_tokens = extract_one(
                client, path, chunk, idx, len(chunks), system
            )
            write_text(dest, f"# {path.name} — chunk {idx}/{len(chunks)}\n\n{text}\n")
            elapsed = time.monotonic() - started
            print(
                f"  wrote {dest.name} in {elapsed:.0f}s max_tokens={max_tokens}",
                flush=True,
            )
            written.append(dest)
    return written


def batched(items: list[Path], size: int) -> Iterable[list[Path]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def estimate_tokens(text: str) -> int:
    """Upper-bound tokens (≈2.2 chars/token). Underestimating prompt 400s llama.cpp."""
    return max(1, (len(text) * 10 + 21) // 22)


def completion_budget(system: str, user: str, cap: int) -> int:
    """n_predict = min(stage cap, leftover n_ctx after prompt and headroom)."""
    prompt = estimate_tokens(system) + estimate_tokens(user)
    room = N_CTX - CTX_HEADROOM - prompt
    if room < 1024:
        raise RuntimeError(
            f"prompt ~{prompt} tokens leaves {room} for completion "
            f"(n_ctx={N_CTX}, headroom={CTX_HEADROOM}). Split the input."
        )
    return min(cap, room)


def prompt_room(system: str, user: str) -> int:
    """Tokens left for completion after this prompt."""
    prompt = estimate_tokens(system) + estimate_tokens(user)
    return N_CTX - CTX_HEADROOM - prompt


def fits_completion(system: str, user: str, cap: int) -> bool:
    return prompt_room(system, user) >= cap


def join_user(prefix: str, paths: list[Path]) -> str:
    parts = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        text = text.replace(TRUNCATION_MARK, "").strip()
        parts.append(text)
    body = "\n\n-----\n\n".join(parts)
    return f"/nothink\n{prefix}\n\n{body}"


def pack_paths(
    paths: list[Path],
    system: str,
    prefix: str,
    min_completion: int,
) -> list[list[Path]]:
    """Group files so each merge keeps at least min_completion output room."""
    groups: list[list[Path]] = []
    current: list[Path] = []
    for path in paths:
        trial = current + [path]
        user = join_user(prefix, trial)
        room = prompt_room(system, user)
        if current and room < min_completion:
            groups.append(current)
            current = [path]
            continue
        current = trial
    if current:
        groups.append(current)
    return groups


def reduce_layer(
    client: LlamaClient,
    sources: list[Path],
    dest_dir: Path,
    stem: str,
    resume: bool,
    system: str,
) -> list[Path]:
    """Merge ``sources`` into batches that leave a large completion budget."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    prefix = "Merge these extracts."
    groups = pack_paths(sources, system, prefix, REDUCE_MAX_TOKENS)
    for group_i, group in enumerate(groups, start=1):
        dest = dest_dir / f"{stem}-{group_i:02d}.md"
        if resume and is_complete(dest):
            print(f"resume {dest.name}", flush=True)
            written.append(dest)
            continue
        user = join_user(prefix, group)
        max_tokens = completion_budget(system, user, REDUCE_MAX_TOKENS)
        print(
            f"{stem} {group_i}/{len(groups)} ({len(group)} files, "
            f"max_tokens={max_tokens})",
            flush=True,
        )
        text = client.chat(system, user, max_tokens=max_tokens)
        write_text(dest, text)
        written.append(dest)
    return written


def try_persona_direct(
    client: LlamaClient,
    sources: list[Path],
    out_dir: Path,
    resume: bool,
    persona_system: str,
) -> Path | None:
    """One PERSONA_SYSTEM call when every source file already fits in n_ctx."""
    persona_prefix = (
        "Write the final mimic-ready persona from these merged extracts. "
        "Follow the output schema in the system prompt."
    )
    probe = join_user(persona_prefix, sources)
    if not fits_completion(persona_system, probe, PERSONA_MAX_TOKENS):
        return None
    persona_path = out_dir / "persona.md"
    if resume and is_complete(persona_path):
        print(f"resume {persona_path}", flush=True)
        return persona_path
    max_tokens = completion_budget(persona_system, probe, PERSONA_MAX_TOKENS)
    print(
        f"final persona synthesis ({len(sources)} files, max_tokens={max_tokens})",
        flush=True,
    )
    persona = client.chat(
        persona_system, probe, max_tokens=max_tokens, temperature=0.3
    )
    write_text(persona_path, persona)
    return persona_path


def run_reduce(
    client: LlamaClient,
    extracts: list[Path],
    out_dir: Path,
    resume: bool,
    reduce_system: str,
    persona_system: str,
) -> Path:
    """One reduce layer when 262k can hold the merge; fold only if it still overflows."""
    layer_sources = sorted(extracts)
    direct = try_persona_direct(
        client, layer_sources, out_dir, resume, persona_system
    )
    if direct is not None:
        return direct
    layer_i = 1
    while True:
        dest_dir = out_dir / "reduce" if layer_i == 1 else out_dir / f"reduce-fold{layer_i}"
        stem = "group" if layer_i == 1 else f"fold{layer_i}"
        prev_n = len(layer_sources)
        layer_sources = reduce_layer(
            client, layer_sources, dest_dir, stem, resume, reduce_system
        )
        direct = try_persona_direct(
            client, layer_sources, out_dir, resume, persona_system
        )
        if direct is not None:
            return direct
        if len(layer_sources) == 1:
            raise RuntimeError(
                "single folded file still too large for persona; "
                "raise --n-ctx or split the corpus."
            )
        if len(layer_sources) >= prev_n:
            raise RuntimeError(
                f"fold did not shrink ({prev_n} → {len(layer_sources)})"
            )
        layer_i += 1
        if layer_i > 8:
            raise RuntimeError("reduce did not fit after 8 folds")


def self_test() -> int:
    """No GPU. Checks chunking, prompt fill, and CoT strip."""
    chunks = chunk_text("a" * 100, size=40, overlap=5)
    assert len(chunks) >= 3, chunks
    assert "gone" in strip_reasoning("pre<think>secret</think>gone")
    assert estimate_tokens("abcd") == 2
    extract = fill_prompt(EXTRACT_SYSTEM, DEFAULT_SHOW, DEFAULT_HOST)
    reduce = fill_prompt(REDUCE_SYSTEM, DEFAULT_SHOW, DEFAULT_HOST)
    persona = fill_prompt(PERSONA_SYSTEM, DEFAULT_SHOW, DEFAULT_HOST)
    assert "The Poz Button" in extract
    assert "Borzoi" in reduce
    assert "Brand wall" in persona
    assert "[[SHOW]]" not in extract + reduce + persona
    room = completion_budget(extract, "user", PERSONA_MAX_TOKENS)
    assert room == PERSONA_MAX_TOKENS
    assert prompt_room(extract, "user") > 250_000
    assert N_CTX == 262_144
    size = extract_chunk_chars(extract)
    assert size > 200_000
    long_ep = "x" * 300_000
    assert len(chunk_text(long_ep, size=size)) == 1
    tmp_dir = Path("/tmp/poz-persona-self-test")
    tmp_dir.mkdir(exist_ok=True)
    cut = tmp_dir / "cut.md"
    cut.write_text(f"hello\n{TRUNCATION_MARK}\n", encoding="utf-8")
    ok = tmp_dir / "ok.md"
    ok.write_text("hello\n", encoding="utf-8")
    assert not is_complete(cut)
    assert is_complete(ok)
    print("self-test ok")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reverse-engineer a film-review podcast persona from transcripts "
            "via llama.cpp on port 8018."
        )
    )
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--show", default=DEFAULT_SHOW, help="Show title to profile")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Primary speaker to profile")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Max transcript files (0 = all)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true", help="Health-check 8018 and exit")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument(
        "--reduce-only",
        action="store_true",
        help="Reuse existing extracts/; skip the transcript map pass",
    )
    parser.add_argument(
        "--redo-truncated",
        action="store_true",
        help="Regenerate extracts that hit max_tokens; wipe reduce/persona",
    )
    parser.add_argument(
        "--persona-from",
        type=Path,
        default=None,
        help="Skip extract/reduce; one-shot persona.md if the dir fits n_ctx",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        help="Override n_ctx (default 262144; must match llama.cpp --ctx-size)",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--only",
        default="",
        help="Substring filter on filename (e.g. 'Liberty Valance')",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    global N_CTX
    args = parse_args(argv)
    if args.n_ctx is not None:
        if args.n_ctx < 8_192:
            print("--n-ctx is too small", file=sys.stderr)
            return 1
        N_CTX = args.n_ctx
    if args.self_test:
        return self_test()

    client = LlamaClient(args.base_url, args.model)
    if args.check:
        ok, msg = client.health()
        print(msg)
        return 0 if ok else 1

    ok, msg = client.health()
    extract_system = fill_prompt(EXTRACT_SYSTEM, args.show, args.host)
    reduce_system = fill_prompt(REDUCE_SYSTEM, args.show, args.host)
    persona_system = fill_prompt(PERSONA_SYSTEM, args.show, args.host)

    if args.persona_from is not None:
        source = args.persona_from
        if not source.is_dir():
            print(f"no fold dir: {source}", file=sys.stderr)
            return 1
        files = sorted(source.glob("*.md"))
        print(f"persona-from: {len(files)} files in {source}")
        print(f"show: {args.show}  host: {args.host}")
        print(f"endpoint: {args.base_url}  model: {args.model}  n_ctx: {N_CTX}")
        if args.dry_run:
            for p in files:
                print(f"  {p.name}")
            return 0
        if not ok:
            print(msg, file=sys.stderr)
            return 1
        if not files:
            print("no markdown in fold dir", file=sys.stderr)
            return 1
        persona_path = try_persona_direct(
            client, files, args.out, resume=args.resume, persona_system=persona_system
        )
        if persona_path is None:
            print(
                "those files still do not fit n_ctx; run reduce first",
                file=sys.stderr,
            )
            return 1
        print(f"persona: {persona_path}")
        return 0

    if args.redo_truncated and args.reduce_only:
        print("use --redo-truncated or --reduce-only, not both", file=sys.stderr)
        return 1

    if args.reduce_only:
        extract_dir = args.out / "extracts"
        extracts = sorted(extract_dir.glob("*.md"))
        incomplete = [p for p in extracts if not is_complete(p)]
        print(f"reduce-only: {len(extracts)} extracts in {extract_dir}")
        print(f"show: {args.show}  host: {args.host}")
        print(f"endpoint: {args.base_url}  model: {args.model}  n_ctx: {N_CTX}")
        if incomplete:
            print(
                f"{len(incomplete)} extracts are still truncated; "
                "run without --reduce-only so they regenerate.",
                file=sys.stderr,
            )
            return 1
        if args.dry_run:
            return 0
        if not ok:
            print(msg, file=sys.stderr)
            return 1
        if not extracts:
            print("no extracts to reduce", file=sys.stderr)
            return 1
        clear_derived(args.out)
        persona_path = run_reduce(
            client,
            extracts,
            args.out,
            resume=False,
            reduce_system=reduce_system,
            persona_system=persona_system,
        )
        print(f"persona: {persona_path}")
        return 0

    if not args.transcripts.is_dir():
        print(f"no transcripts dir: {args.transcripts}", file=sys.stderr)
        return 1

    files = iter_transcripts(args.transcripts)
    if args.only:
        needle = args.only.lower()
        files = [p for p in files if needle in p.name.lower()]
    if args.limit > 0:
        files = files[: args.limit]

    chunk_size = extract_chunk_chars(extract_system)
    chunk_counts = []
    for path in files:
        n = len(chunk_text(load_text(path), size=chunk_size))
        chunk_counts.append((n, path.name))

    print(f"files: {len(files)}  chunks: {sum(n for n, _ in chunk_counts)}")
    print(f"extract chunk size: {chunk_size} chars")
    print(f"show: {args.show}  host: {args.host}")
    print(f"out: {args.out}")
    print(f"endpoint: {args.base_url}  model: {args.model}  n_ctx: {N_CTX}")
    print("this stack is --parallel 1; a full run is hours. use --resume.")
    if args.dry_run:
        for n, name in chunk_counts:
            print(f"  {n:2d}  {name}")
        return 0

    if not ok:
        print(msg, file=sys.stderr)
        return 1
    print(f"health: {msg}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    extract_dir = args.out / "extracts"
    incomplete = (
        [p for p in extract_dir.glob("*.md") if not is_complete(p)]
        if extract_dir.exists()
        else []
    )
    if incomplete:
        print(
            f"will regenerate {len(incomplete)} truncated extracts; "
            "clearing reduce/persona",
            flush=True,
        )
        clear_derived(args.out)
    extracts = run_extracts(
        client, files, args.out, resume=args.resume, system=extract_system
    )
    print(f"extracts ready: {len(extracts)}", flush=True)
    if args.extract_only:
        print(f"extracts: {len(extracts)} under {args.out / 'extracts'}")
        return 0
    if not extracts:
        print("no extracts to reduce", file=sys.stderr)
        return 1
    persona = run_reduce(
        client,
        extracts,
        args.out,
        resume=args.resume,
        reduce_system=reduce_system,
        persona_system=persona_system,
    )
    print(f"persona: {persona}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
