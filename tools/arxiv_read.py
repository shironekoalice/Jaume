import urllib.request, urllib.parse, xml.etree.ElementTree as ET, json, time

NS = {"a": "http://www.w3.org/2005/Atom"}

def by_ids(ids):
    out = []
    for i in range(0, len(ids), 25):
        chunk = ids[i:i + 25]
        url = ("http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)}))
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
        root = ET.fromstring(data)
        for e in root.findall("a:entry", NS):
            aid = e.find("a:id", NS).text.strip().rsplit("/", 1)[-1]
            base = aid.split("v")[0]
            out.append({
                "id": base, "version": aid,
                "title": " ".join(e.find("a:title", NS).text.split()),
                "published": e.find("a:published", NS).text[:10],
                "updated": e.find("a:updated", NS).text[:10],
                "cats": sorted({c.get("term") for c in e.findall("a:category", NS)}),
                "authors": [a.find("a:name", NS).text
                            for a in e.findall("a:author", NS)],
                "abstract": " ".join(e.find("a:summary", NS).text.split()),
                "comment": (" ".join(e.find("{http://arxiv.org/schemas/atom}comment").text.split())
                            if e.find("{http://arxiv.org/schemas/atom}comment") is not None else ""),
            })
        time.sleep(3.2)
    return out

KEY = [
    "2606.19348",   # DeepSeek-V4
    "2512.02556",   # DeepSeek-V3.2
    "2609.19969",   # DeepSeek-V4.1-Flash
    "2610.24766",   # ECHO: causal self-recursion
    "2608.03624",   # LoopMTP
    "2607.15456",   # Looped Latent Attention
    "2604.11912",   # How Transformers Learn to Plan via MTP
    "2607.20145",   # SLAI T-Rex on Ascend
    "2611.03768",   # Multi-layer Indexer (DSA)
    "2412.19437",   # DeepSeek-V3
    "2501.12948",   # DeepSeek-R1
    "2607.23054",   # MLA separates content from position
    "2601.11464",   # MHA2MLA-VLM
    "2601.20552",   # DeepSeek-OCR 2
    "2610.38331",   # Multi-Token Prediction Done Right
    "2610.28619",   # Adaptive Multi-Token Prediction
    "2607.29279",   # ParaASR MTP
    "2604.06155",   # world models with MTP
    "2602.06019",   # MTP via Self-Distillation
    "2609.07008",   # RedKnot-MLA long context
]
recs = by_ids(KEY)
print(f"fetched {len(recs)}/{len(KEY)}")
for r in recs:
    print("=" * 100)
    print(f"[{r['id']}] {r['published']}  (updated {r['updated']})")
    print(f"TITLE: {r['title']}")
    print(f"AUTHORS: {', '.join(r['authors'][:8])}")
    print(f"CATS: {', '.join(r['cats'])}")
    if r["comment"]:
        print(f"COMMENT: {r['comment'][:300]}")
    print(f"ABSTRACT: {r['abstract']}")
json.dump(recs, open("docs/.deepseek_key.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
