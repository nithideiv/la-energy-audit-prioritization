from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://data.lacity.org/api/views/9yda-i4ya/rows.csv?accessType=DOWNLOAD"
OUT = ROOT / "data/raw/ebewe_all.csv"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(OUT, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Saved {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

