import json
recs = json.load(open("docs/.deepseek_key.json", encoding="utf-8"))
by = {r["id"]: r for r in recs}
for w in ["2501.12948","2412.19437","2608.03624","2607.20145","2610.38331"]:
    r = by.get(w)
    if not r:
        print(f"### {w}: NOT FETCHED"); continue
    print("=" * 90)
    print(f"[{r['id']}] {r['published']}  upd {r['updated']}")
    print(f"TITLE: {r['title']}")
    print(f"AUTHORS: {', '.join(r['authors'][:6])}")
    if r["comment"]: print(f"COMMENT: {r['comment'][:200]}")
    print(f"ABSTRACT: {r['abstract'][:2100]}")
