import argparse
import json
import os
import sys
from dataclasses import replace

from youtube_api import DEFAULT_CONFIG, find_best_youtube_segment, is_cooking_video_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test YouTube recipe-video recommendation without loading the LLM server."
    )
    parser.add_argument("query", help="Search query, for example: 김치볶음밥 만드는 법 영상")
    parser.add_argument("--api-key", default=os.getenv("YOUTUBE_API_KEY"), help="YouTube Data API key")
    parser.add_argument("--max-results", type=int, default=5, help="Number of YouTube search results to inspect")
    parser.add_argument("--top-segments", type=int, default=3, help="Number of transcript segments to return")
    parser.add_argument(
        "--asr",
        action="store_true",
        help="Enable Whisper ASR fallback when transcripts are unavailable. This is slower.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("YOUTUBE_API_KEY is required. Set it in the environment or pass --api-key.", file=sys.stderr)
        return 2

    config = replace(
        DEFAULT_CONFIG,
        max_results=args.max_results,
        top_segments=args.top_segments,
        enable_asr_fallback=args.asr,
        asr_max_videos=1 if args.asr else 0,
    )

    print(f"cooking_video_query={is_cooking_video_query(args.query)}")
    result = find_best_youtube_segment(args.query, args.api_key, config=config)
    if result is None:
        print("No recommendation found.")
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
