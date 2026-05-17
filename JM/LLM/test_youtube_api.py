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
    parser.add_argument("--max-results", type=int, default=2, help="Number of top YouTube search results to inspect")
    parser.add_argument("--top-segments", type=int, default=3, help="Number of transcript segments to return")
    parser.add_argument(
        "--asr",
        action="store_true",
        help="Keep Whisper ASR fallback enabled. This is the default and is kept for compatibility.",
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        help="Disable Whisper ASR fallback and only use existing YouTube transcripts.",
    )
    parser.add_argument(
        "--allow-full-video",
        action="store_true",
        help="Return a title/description match from 0 seconds when no timeline segment is found.",
    )
    return parser.parse_args()


def main() -> int:
    # 윈도우 환경에서 이모지 등 출력 시 cp949 인코딩 에러 방지
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if not args.api_key:
        print("YOUTUBE_API_KEY is required. Set it in the environment or pass --api-key.", file=sys.stderr)
        return 2

    config = replace(
        DEFAULT_CONFIG,
        max_results=args.max_results,
        top_segments=args.top_segments,
        allow_metadata_fallback=args.allow_full_video,
        enable_asr_fallback=not args.no_asr,
        asr_max_videos=args.max_results if not args.no_asr else 0,
    )

    print(f"cooking_video_query={is_cooking_video_query(args.query)}")
    print(f"search_top_videos={config.max_results}")
    print(f"asr_fallback={config.enable_asr_fallback}")
    result = find_best_youtube_segment(args.query, args.api_key, config=config)
    if result is None:
        print("No timeline segment found.")
        print("Try --asr for Whisper-based timeline search, or --allow-full-video if a 0-second video link is acceptable.")
        return 1

    if not result.get("timeline_found", False):
        print("Warning: this is a full-video fallback, not a timeline match.")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
