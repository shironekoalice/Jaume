import urllib.request, urllib.parse, xml.etree.ElementTree as ET, json, time, sys

NS = {"a": "http://www.w3.org/2005/Atom"}
API = "http://export.arxiv.org/api/query"

def q(search, max_results=60, sort="submittedDate"):
    url = API + "?" + urllib.parse.urlencode({
        "search_query": search, "start": 0, "max_results": max_results,
        "sortBy": sort, "sortOrder": "descending"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(5)
    root = ET.fromstring(data)
    out = []
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.strip().rsplit("/", 1)[-1]
        base = aid.split("v")[0]
        out.append({
            "id": base,
            "version": aid,
            "title": " ".join(e.find("a:title", NS).text.split()),
            "published": e.find("a:published", NS).text[:10],
            "updated": e.find("a:updated", NS).text[:10],
            "cats": sorted({c.get("term") for c in e.findall("a:category", NS)}),
            "authors": [a.find("a:name", NS).text
                        for a in e.findall("a:author", NS)],
            "abstract": " ".join(e.find("a:summary", NS).text.split()),
        })
    return out

QUERIES = [
    'ti:"DeepSeek"',
    'au:"DeepSeek-AI"',
    'ti:"Multi-Token Prediction"',
    'ti:"Multi-head Latent Attention"',
    'ti:"DeepSeek Sparse Attention"',
    'ti:"latent attention" AND cat:cs.LG',
    'ti:"mixture-of-experts" AND ti:"DeepSeek"',
]

seen = {}
for s in QUERIES:
    try:
        res = q(s)
    except Exception as e:
        print(f"QUERY FAILED {s}: {e}", file=sys.stderr)
        continue
    print(f"### {s}  -> {len(res)}")
    for r in res:
        if r["id"] not in seen:
            r["found_by"] = [s]
            seen[r["id"]] = r
        else:
            seen[r["id"]]["found_by"].append(s)
    time.sleep(3.2)

print(f"\nTOTAL UNIQUE: {len(seen)}")
recs = sorted(seen.values(), key=lambda r: r["published"], reverse=True)
for r in recs:
    print(f"{r['id']:14s} {r['published']}  {r['title'][:92]}")
json.dump(recs, open("docs/.deepseek_raw.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
