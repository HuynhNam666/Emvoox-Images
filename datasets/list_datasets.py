"""Liệt kê dataset theo nhóm. Không tải gì — chỉ in URL/license/cách truy cập."""
import json
import pathlib
import sys

reg = json.loads((pathlib.Path(__file__).parent / "registry.json").read_text(encoding="utf-8"))
group = sys.argv[1] if len(sys.argv) > 1 else None
for g, names in reg["groups"].items():
    if group and g != group:
        continue
    print(f"\n[{g}]")
    for n in names:
        d = reg["datasets"][n]
        flag = "OK " if d["verified"] else "?? "
        print(f"  {flag}{n:15} {d['license']:32} {d['access']:10} {d['url']}")
