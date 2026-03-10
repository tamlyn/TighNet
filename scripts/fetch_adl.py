"""Download and extract the ADL Piano MIDI dataset."""

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/lucasnfe/adl-piano-midi/raw/refs/heads/master/midi/adl-piano-midi.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "adl-piano-midi"


def main() -> None:
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.mid")):
        count = sum(1 for _ in DATA_DIR.rglob("*.mid"))
        print(f"Already have {count} MIDI files in {DATA_DIR}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {URL} ...")
    response = urllib.request.urlopen(URL)
    data = response.read()
    print(f"Downloaded {len(data) / 1024 / 1024:.1f} MB")

    print(f"Extracting to {DATA_DIR} ...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(DATA_DIR)

    count = sum(1 for _ in DATA_DIR.rglob("*.mid"))
    print(f"Done — {count} MIDI files extracted")


if __name__ == "__main__":
    main()
