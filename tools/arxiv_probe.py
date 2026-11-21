import urllib.request, urllib.parse, xml.etree.ElementTree as ET, json, sys, time

NS = {"a": "http://www.w3.org/2005/Atom"}

def q(search, max_results=100, sort="submittedDate"):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": search, "start": 0, "max_results": max_results,
        "sortBy": sort, "sortOrder": "descending"})
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    root = ET.fromstring(data)
    out = []
    for e in root.findall("a:entry", NS):
        idu = e.find("a:id", NS).text.strip()
        aid = idu.rsplit("/", 1)[-1]
        title = " ".join(e.find("a:title", NS).text.split())
        pub = e.find("a:published", NS).text[:10]
        upd = e.find("a:updated", NS).text[:10]
        summ = " ".join(e.find("a:summary", NS).text.split())
        cats = [c.get("term") for c in e.findall("a:category", NS)]
        authors = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
        out.append({"id": aid, "title": title, "published": pub,
                    "updated": upd, "cats": cats, "authors": authors,
                    "abstract": summ})
    return out

if __name__ == "__main__":
    res = q("all:DeepSeek", 100)
    print("QUERY all:DeepSeek ->", len(res))
    for r in res[:40]:
        print(f"  {r['id']:14s} {r['published']}  {r['title'][:88]}")
