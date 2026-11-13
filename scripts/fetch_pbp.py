"""Download NFL play-by-play data (the public nflverse release) into data/raw/pbp/.

    python scripts/fetch_pbp.py 2024 2026

Each season is one file. The current season's file is refreshed nightly upstream, so
rerun for it whenever new games have been played. Standard library only.
"""
import os
import sys
import urllib.request

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
OUT = "data/raw/pbp"


def main(first: int, last: int) -> None:
    os.makedirs(OUT, exist_ok=True)
    for season in range(first, last + 1):
        path = os.path.join(OUT, f"play_by_play_{season}.csv.gz")
        request = urllib.request.Request(URL.format(season=season), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=300) as response, open(path, "wb") as fh:
            fh.write(response.read())
        print(f"{season}: {os.path.getsize(path) / 1e6:.1f} MB -> {path}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else int(sys.argv[1]))
