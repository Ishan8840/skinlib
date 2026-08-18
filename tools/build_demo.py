"""Build the app demo page from the real render bundle."""

import json
from pathlib import Path

ROOT = Path("/home/ishan/projects/acne")
OUT = Path("data/field-test.html")
cards = {c["name"]: c for c in json.loads((ROOT / "data/demo.json").read_text())}

TEMPLATE = Path(__file__).with_name("demo_template.html").read_text()
payload = json.dumps(
    [
        {
            "id": key,
            "who": meta["who"],
            "sub": meta["sub"],
            "photo": cards[key]["photo"],
            "overlay": cards[key]["overlay"],
            "zones": cards[key]["zones"],
            "count": round(cards[key]["estimated_count"]),
            "flags": cards[key]["flags"],
            "face_px": cards[key]["face_px"],
        }
        for key, meta in {
            "personB": {"who": "Ravi", "sub": "iPhone 15 Pro · indoor, overhead light"},
            "personA": {"who": "Dev", "sub": "iPhone 15 Pro · window light from the left"},
            "levle0_203": {
                "who": "Held-out case",
                "sub": "clinic camera · never seen in training",
            },
        }.items()
    ],
    separators=(",", ":"),
)
OUT.write_text(TEMPLATE.replace("/*__DATA__*/null", payload))
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
