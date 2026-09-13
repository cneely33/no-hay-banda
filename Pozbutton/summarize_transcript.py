#!/usr/bin/env python3
"""
Summarize the movie review in one podcast transcript via llama.cpp on port 8018.

This is a recap of what the hosts argued, not a new review and not a plot dump.

Usage:
  python3 summarize_transcript.py --check
  python3 summarize_transcript.py --dry-run --transcript transcripts/The\\ Poz\\ Button\\ 24\\ -\\ Drive.txt
  python3 summarize_transcript.py --transcript path/to/episode.txt --words 400
  python3 summarize_transcript.py --transcripts /usr/src/no-hay-banda/transcripts --words 400
  python3 summarize_transcript.py --transcripts transcripts --out-dir transcripts/summaries --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import build_persona as bp

DEFAULT_WORDS = 400
MIN_WORDS = 80
MAX_WORDS = 2_000
CHUNK_SUMMARY_WORDS = 220
DEFAULT_SUMMARY_SUBDIR = "summaries"


def words_to_tokens(word_count: int) -> int:
    """Completion cap with headroom so the model can hit the word target."""
    return max(256, word_count * 3)


def summary_system(word_count: int, partial: bool) -> str:
    role = (
        "This is one slice of a longer transcript. Extract the review claims "
        "in this slice only. Do not write the final overall summary yet."
        if partial
        else "Write one overall summary of the movie review in this transcript."
    )
    return f"""\
You summarize a film-review podcast transcript.

{role}

Target length: about {word_count} words (not characters). Stay within 15% of that.

Content:
- Name the film or films under review if the hosts do.
- Cover only films the hosts actually review. A film named for comparison, context, or as an aside is not under review: mention it only inside the argument it supports, and do not give it its own verdict or section. When several films are genuinely reviewed, handle each separately in the order the hosts take them up.
- State the hosts' verdict and the main arguments, including guest disagreements.
- Prefer the review (what they thought and why) over plot recap.
- Do not impersonate the hosts. Third person: "the host argues…"
- Do not reproduce slurs; paraphrase if a slur is load-bearing to the take.
- Skip running gags, tangents, and bits that carry no evaluation.
- Read sarcasm and irony for intent, not literally: hosts often praise in mock-outrage or pan in mock-enthusiasm, and a literal read inverts the verdict
- Do not invent claims that are not in this text.
- Do not editorialize the hosts' arguments. 
- Do present the arguments as they are, without adding your own opinions or interpretations.
"""


def summary_path(out_dir: Path, transcript: Path) -> Path:
    """One markdown file per transcript, named from the transcript stem."""
    return out_dir / f"{bp.slug(transcript.stem)}.md"


def collect_transcripts(path: Path) -> list[Path]:
    """A single .txt file, or every nested .txt under a directory."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return bp.iter_transcripts(path)
    return []


def is_complete_summary(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return bp.TRUNCATION_MARK not in text


def chunk_size_for(system: str, word_count: int) -> int:
    cap = words_to_tokens(word_count)
    reserved = bp.CTX_HEADROOM + cap + bp.estimate_tokens(system) + 400
    token_room = max(bp.N_CTX - reserved, 4_096)
    return max(20_000, (token_room * 22) // 10)


def summarize_text(
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


def run_summary(
    client: bp.LlamaClient,
    transcript: Path,
    word_count: int,
    temperature: float,
) -> str:
    text = bp.load_text(transcript)
    if not text.strip():
        raise ValueError(f"empty transcript: {transcript}")
    final_system = summary_system(word_count, partial=False)
    size = chunk_size_for(final_system, word_count)
    chunks = bp.chunk_text(text, size=size)
    print(f"{transcript.name}: {len(text)} chars, {len(chunks)} chunk(s)", flush=True)
    if len(chunks) == 1:
        return summarize_text(
            client,
            final_system,
            f"Transcript: {transcript.name}",
            chunks[0],
            word_count,
            temperature,
        )

    partial_system = summary_system(min(CHUNK_SUMMARY_WORDS, word_count), partial=True)
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"chunk {idx}/{len(chunks)}", flush=True)
        part = summarize_text(
            client,
            partial_system,
            f"Transcript: {transcript.name} — chunk {idx}/{len(chunks)}",
            chunk,
            min(CHUNK_SUMMARY_WORDS, word_count),
            temperature,
        )
        parts.append(f"## Chunk {idx}/{len(chunks)}\n\n{part.strip()}")
    merged = "\n\n".join(parts)
    print("folding chunk summaries into one overall summary", flush=True)
    return summarize_text(
        client,
        final_system,
        f"Transcript: {transcript.name}. Merge these partial summaries "
        f"into one overall summary of about {word_count} words.",
        merged,
        word_count,
        temperature,
    )


def self_test() -> int:
    assert words_to_tokens(400) >= 1_200
    sys_prompt = summary_system(400, partial=False)
    assert "400 words" in sys_prompt
    assert "overall summary" in sys_prompt
    size = chunk_size_for(sys_prompt, 400)
    assert size >= 20_000
    chunks = bp.chunk_text("a" * 100, size=40, overlap=5)
    assert len(chunks) >= 3
    dest = summary_path(Path("/tmp/summaries"), Path("The Poz Button 24 - Drive.txt"))
    assert dest.name == "The_Poz_Button_24_-_Drive.md"
    print("self-test ok")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the movie review in one transcript, or every .txt "
            "in a directory, via llama.cpp on 8018."
        )
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="One .txt transcript, or a directory of transcripts",
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=None,
        help="Directory of .txt transcripts (same as --transcript DIR)",
    )
    parser.add_argument(
        "--words",
        "--length",
        dest="words",
        type=int,
        default=DEFAULT_WORDS,
        help=f"Target length of the overall summary in words (default {DEFAULT_WORDS})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file for a single transcript (ignored in directory mode)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory for one summary file per transcript. Default in "
            f"directory mode: <transcripts>/{DEFAULT_SUMMARY_SUBDIR}"
        ),
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

    source = args.transcripts or args.transcript
    if source is None:
        print("need --transcript or --transcripts", file=sys.stderr)
        return 1
    files = collect_transcripts(source)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"no .txt transcripts in {source}", file=sys.stderr)
        return 1

    batch = source.is_dir() or len(files) > 1
    out_dir = args.out_dir
    if batch and out_dir is None:
        root = source if source.is_dir() else source.parent
        out_dir = root / DEFAULT_SUMMARY_SUBDIR

    system = summary_system(args.words, partial=False)
    size = chunk_size_for(system, args.words)
    print(f"files: {len(files)}  chunk_size: {size}  target words: {args.words}")
    print(f"endpoint: {args.base_url}")
    if out_dir:
        print(f"out-dir: {out_dir}")
    for path in files:
        n_chunks = len(bp.chunk_text(bp.load_text(path), size=size))
        dest = summary_path(out_dir, path) if out_dir else args.out
        print(f"  {n_chunks:2d} chunks  {path.name}" + (f" -> {dest.name}" if dest else ""))
    if args.dry_run:
        return 0

    ok, msg = client.health()
    if not ok:
        print(msg, file=sys.stderr)
        return 1

    if batch:
        assert out_dir is not None
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        skipped = 0
        for file_i, path in enumerate(files, start=1):
            dest = summary_path(out_dir, path)
            if args.resume and is_complete_summary(dest):
                print(f"[{file_i}/{len(files)}] resume {dest.name}", flush=True)
                skipped += 1
                continue
            print(f"[{file_i}/{len(files)}] {path.name}", flush=True)
            summary = run_summary(client, path, args.words, args.temperature)
            bp.write_text(dest, summary.strip() + "\n")
            print(f"  wrote {dest}", flush=True)
            written += 1
        print(f"done: wrote {written}, skipped {skipped}, out-dir {out_dir}")
        return 0

    path = files[0]
    summary = run_summary(client, path, args.words, args.temperature)
    print(summary)
    dest = args.out
    if dest is None and out_dir is not None:
        dest = summary_path(out_dir, path)
    if dest:
        bp.write_text(dest, summary.strip() + "\n")
        print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
