"""Download weekly DraftKings results (points and salary per player) from RotoGuru.

    python scripts/fetch_rotoguru.py 2017 2025

Writes data/history/dk_results.csv. Standard library only. The site is plain HTTP, so
run this on a machine with ordinary internet access. Pages are cached in
data/raw/rotoguru/, so a rerun only fetches what is missing.
"""
import csv
import html
import io
import os
import re
import sys
import time
import urllib.request

URL = "http://rotoguru1.com/cgi-bin/fyday.pl?game=dk&scsv=1&week={week}&year={year}"
CACHE = "data/raw/rotoguru"
OUT = "data/history/dk_results.csv"
COLUMNS = ["Week", "Year", "GID", "Name", "Pos", "Team", "h/a", "Oppt", "DK points", "DK salary"]


def parse(page: str) -> list[dict]:
    """Rows from the semicolon-separated block inside the page's <pre> tag."""
    block = re.search(r"(?is)<pre[^>]*>(.*?)</pre>", page)
    if not block:
        return []
    text = html.unescape(re.sub(r"<[^>]+>", "", block.group(1))).strip()
    rows = list(csv.DictReader(io.StringIO(text), delimiter=";"))
    return [r for r in rows if r.get("Name") and r.get("Week", "").strip().isdigit()]


def fetch(year: int, week: int) -> str:
    path = os.path.join(CACHE, f"{year}_{week:02d}.html")
    if os.path.exists(path):
        with open(path, encoding="latin-1") as fh:
            return fh.read()
    request = urllib.request.Request(URL.format(week=week, year=year), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        page = response.read().decode("latin-1")
    time.sleep(3)  # be polite to the site
    if parse(page):  # do not cache weeks that have not been played yet
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "w", encoding="latin-1") as fh:
            fh.write(page)
    return page


def main(first: int, last: int) -> None:
    rows = []
    for year in range(first, last + 1):
        weeks = 0
        for week in range(1, 19):
            found = parse(fetch(year, week))
            if not found:
                continue
            rows += found
            weeks += 1
        print(f"{year}: {weeks} weeks")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} rows to {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else int(sys.argv[1]))
