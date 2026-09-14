"""
Rewrite one-sided film-review text into dual-reader sample analyses.

The diversity pipeline hunts sold meaning (last image vs billed plot, who
the pictures like, who is the joke, who never pays) and then writes notes
a liker and a hater can both accept as "yes, that is the film." Source
reviews often keep that hunt but wear a team jersey. This script keeps the
claims, scenes, and method; it strips recruitment, scolding, and in-group
voice so the files can be used as few-shot examples.

Same llama.cpp stack as the other Pozbutton scripts (port 8018).

Usage:
  python3 convert_reviews.py --check
  python3 convert_reviews.py --dry-run --input-dir summaries
  python3 convert_reviews.py --input-dir summaries --out-dir pipeline-examples
  python3 convert_reviews.py --input-dir summaries --limit 1 --no-resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import build_persona as bp

DEFAULT_INPUT = Path("/usr/src/no-hay-banda/Pozbutton/summaries")
DEFAULT_OUT_DIR = Path("/usr/src/no-hay-banda/Pozbutton/pipeline-examples")
REVIEW_EXTENSIONS = (".md", ".txt")
DEFAULT_WORDS = 450
MIN_WORDS = 120
MAX_WORDS = 1_200
CHUNK_REWRITE_WORDS = 280
REWRITE_MAX_TOKENS_CAP = 2_048

REWRITE_SYSTEM = """\
You rewrite an existing film analysis into a dual-reader sample for a
movie-guide scoring pipeline. You are not reviewing the film yourself.

Method to keep (this is the value of the source):
- retain the structure and methodology of how a film is critiqued, deconstructed and analyzed

Content rules:
- Keep every load-bearing claim, scene, name, and reading from the source.
- Do not invent credits, scenes, lawsuits, or symbols the source does not
  state. If the source is unsure, keep the uncertainty.
- If the source reviews several films, keep each film separate, in order.
- Third person about the film. Do not impersonate the source hosts.
- Do not name the source show, hosts, guests, or "the hosts argue."
- Write as if the pictures themselves are the evidence.
- Rewrite spoilers: Write as a first-watch adult who has not
  been told the secret. Do not explicitly name the twist.
  Words like "reveals," "it turns out," "the twist is," "in the end,"
  "actually," and "the narrative reveals that" are almost always a
  spoiler dump — cut them and the clause that follows.
  If the source's reading depends on a hidden mechanism, keep the
  reading in setup and tone only: billed visit, who the camera likes,
  who feels like the joke, hospitality that starts to feel like a
  procedure. Do not say what the procedure is.

Jersey to remove (tone only):
- Team labels as the writer's identity: we/they, left/right, woke/based,
  normies, "alt-right cinema," recruiting the reader to a camp
- Cheering or scolding ("this is why they hate it," "you are supposed to")
- Conspiracy asides that are not a claim about a picture, costume, line,
  or last image
- Slurs. If a slur is load-bearing to a claim about what is on screen,
  paraphrase the claim without reproducing the slur
- Theory-word dumps (agency, -normative, validates, celebrates) when a
  picture will do: "she decides," "no gay couple is shown"

Voice:
- Active voice. Name the scene, not the lecture.
- Dual-reader: a liker and a hater must both say "yes, that is the film."
- No "sells a vision," "the film's thesis is," or "this ending inverts"
  as an opener. Open on the billed title or a picture.
- If billed plot and last image disagree, the last image is the argument.
- If no argument is on screen, write entertainment and stop.
- Kakusu: hide the argument in staging. Do not announce a worldview. A liker must not feel lectured; a hater must not feel recruited. Both should recognize the same props.

Length: about [[WORDS]] words (not characters). Stay within 15%.

Output markdown only:

# [Film title] ([year if the source gives one])

## Rewritten Analysis

[Rewritten analysis text]
"""


def words_to_tokens(word_count: int) -> int:
    """Completion cap with headroom so the model can hit the word target."""
    return max(256, min(REWRITE_MAX_TOKENS_CAP, word_count * 3))


def rewrite_system(word_count: int) -> str:
    return REWRITE_SYSTEM.replace("[[WORDS]]", str(word_count))


def collect_reviews(path: Path) -> list[Path]:
    """One review file, or every nested .md/.txt under a directory."""
    if path.is_file():
        suffix = path.suffix.lower()
        return [path] if suffix in REVIEW_EXTENSIONS else []
    if path.is_dir():
        files = [
            child
            for child in path.rglob("*")
            if child.is_file() and child.suffix.lower() in REVIEW_EXTENSIONS
        ]
        return sorted(files)
    return []


def example_path(out_dir: Path, source: Path) -> Path:
    return out_dir / f"{bp.slug(source.stem)}.example.md"


def is_complete_example(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return bp.TRUNCATION_MARK not in text


def chunk_size_for(system: str, word_count: int) -> int:
    cap = words_to_tokens(word_count)
    reserved = bp.CTX_HEADROOM + cap + bp.estimate_tokens(system) + 400
    token_room = max(bp.N_CTX - reserved, 4_096)
    return max(20_000, (token_room * 22) // 10)


def rewrite_text(
    client: bp.LlamaClient,
    system: str,
    header: str,
    body: str,
    word_count: int,
    temperature: float,
) -> str:
    user = f"/nothink\n{header}\n\n{body}"
    max_tokens = bp.completion_budget(system, user, words_to_tokens(word_count))
    print(f"  max_tokens={max_tokens}  target_words={word_count}", flush=True)
    return client.chat(system, user, max_tokens=max_tokens, temperature=temperature)


def convert_one(
    client: bp.LlamaClient,
    source: Path,
    word_count: int,
    temperature: float,
) -> str:
    text = bp.load_text(source)
    if not text.strip():
        raise ValueError(f"empty review: {source}")
    final_system = rewrite_system(word_count)
    size = chunk_size_for(final_system, word_count)
    chunks = bp.chunk_text(text, size=size)
    print(f"{source.name}: {len(text)} chars, {len(chunks)} chunk(s)", flush=True)
    if len(chunks) == 1:
        return rewrite_text(
            client,
            final_system,
            f"Source file: {source.name}\nRewrite this analysis. Keep claims. Remove jersey.",
            chunks[0],
            word_count,
            temperature,
        )

    partial_system = rewrite_system(min(CHUNK_REWRITE_WORDS, word_count))
    parts: list[str] = []
    chunk_total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        print(f"chunk {idx}/{chunk_total}", flush=True)
        part = rewrite_text(
            client,
            partial_system,
            (
                f"Source file: {source.name} — slice {idx}/{chunk_total}. "
                "Rewrite this slice only. Keep claims. Remove jersey. "
                "Do not write the final merged sample yet."
            ),
            chunk,
            min(CHUNK_REWRITE_WORDS, word_count),
            temperature,
        )
        parts.append(f"## Slice {idx}/{chunk_total}\n\n{part.strip()}")
    merged = "\n\n".join(parts)
    print("folding slice rewrites into one sample", flush=True)
    return rewrite_text(
        client,
        final_system,
        (
            f"Source file: {source.name}. Merge these jersey-stripped slices "
            f"into one dual-reader sample of about {word_count} words. "
            "Keep every load-bearing claim. Do not put the jersey back."
        ),
        merged,
        word_count,
        temperature,
    )


def self_test() -> int:
    assert words_to_tokens(450) >= 1_200
    system = rewrite_system(450)
    assert "450" in system
    assert "Dual-reader" in system
    dest = example_path(Path("/tmp/examples"), Path("The_Poz_Button_24_-_Drive.md"))
    assert dest.name == "The_Poz_Button_24_-_Drive.example.md"
    files = collect_reviews(Path("/usr/src/no-hay-banda/Pozbutton/summaries"))
    assert any(path.name.startswith("The_Poz_Button_24") for path in files)
    print("self-test ok")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite film-review text files into dual-reader diversity "
            "pipeline example samples."
        )
    )
    parser.add_argument(
        "--input-dir",
        "--reviews",
        dest="input_dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="One .md/.txt file, or a folder of them",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for one .example.md per source file",
    )
    parser.add_argument(
        "--words",
        type=int,
        default=DEFAULT_WORDS,
        help=f"Target length in words (default {DEFAULT_WORDS})",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max files (0 = all)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--base-url", default=bp.DEFAULT_BASE_URL)
    parser.add_argument("--model", default=bp.DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()

    if args.words < MIN_WORDS or args.words > MAX_WORDS:
        print(
            f"--words must be between {MIN_WORDS} and {MAX_WORDS}",
            file=sys.stderr,
        )
        return 1

    client = bp.LlamaClient(args.base_url, args.model)
    if args.check:
        ok, msg = client.health()
        print(msg)
        return 0 if ok else 1

    files = collect_reviews(args.input_dir)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"no .md/.txt reviews in {args.input_dir}", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    system = rewrite_system(args.words)
    size = chunk_size_for(system, args.words)
    print(f"files: {len(files)}  chunk_size: {size}  target words: {args.words}")
    print(f"endpoint: {args.base_url}")
    print(f"out-dir: {out_dir}")
    for path in files:
        n_chunks = len(bp.chunk_text(bp.load_text(path), size=size))
        dest = example_path(out_dir, path)
        print(f"  {n_chunks:2d} chunks  {path.name} -> {dest.name}")
    if args.dry_run:
        return 0

    ok, msg = client.health()
    if not ok:
        print(msg, file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for file_i, path in enumerate(files, start=1):
        dest = example_path(out_dir, path)
        if args.resume and is_complete_example(dest):
            print(f"[{file_i}/{len(files)}] resume {dest.name}", flush=True)
            skipped += 1
            continue
        print(f"[{file_i}/{len(files)}] {path.name}", flush=True)
        sample = convert_one(client, path, args.words, args.temperature)
        bp.write_text(dest, sample.strip() + "\n")
        print(f"  wrote {dest}", flush=True)
        written += 1
    print(f"done: wrote {written}, skipped {skipped}, out-dir {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))