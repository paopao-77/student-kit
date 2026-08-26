from pathlib import Path


root = Path("data/raw/PHEME/all-rnr-annotated-threads")
rows = []
for event_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("._")):
    for label_name in ["rumours", "non-rumours"]:
        label_dir = event_dir / label_name
        if not label_dir.exists():
            continue
        for thread_dir in sorted(p for p in label_dir.iterdir() if p.is_dir() and not p.name.startswith("._")):
            reactions = thread_dir / "reactions"
            n = len([p for p in reactions.glob("*.json") if not p.name.startswith("._")]) if reactions.exists() else 0
            rows.append((n, event_dir.name, label_name, thread_dir.name))

for n, event, label, tid in sorted(rows, reverse=True)[:30]:
    print(f"{n}\t{event}\t{label}\t{tid}")

print(f"threads={len(rows)}")
