import json, sys
recs = json.load(open("docs/.deepseek_key.json", encoding="utf-8"))
want = ["2606.19348","2512.02556","2610.24766","2608.03624","2607.20145",
        "2501.12948","2412.19437","2610.38331","2610.28619","2601.20552",
        "2611.03768"]
by = {r["id"]: r for r in recs}
for w in want:
    r = by.get(w)
    if not r:
        print(f"### {w}: NOT FETCHED"); continue
    print("=" * 96)
    print(f"[{r['id']}] {r['published']}  upd {r['updated']}")
    print(f"TITLE: {r['title']}")
    print(f"AUTHORS: {', '.join(r['authors'][:6])}")
    print(f"CATS: {', '.join(r['cats'])}")
    if r["comment"]: print(f"COMMENT: {r['comment'][:240]}")
    print(f"ABSTRACT: {r['abstract']}")
