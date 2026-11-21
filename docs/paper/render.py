"""Render paper.tex into a readable standalone HTML (works offline).

Reading copy only; paper.tex remains the authoritative source.
Handles the LaTeX subset actually used there: sections, paragraphs, abstract,
figures, tables, equations, refs/eqrefs, cites, and common inline macros.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "paper.tex")
OUT = os.path.join(HERE, "paper.html")

TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>@@TITLE@@</title>
<link rel="stylesheet" href="paper.css">
<script>
window.MathJax = {
  tex: {inlineMath: [['$','$']], displayMath: [['$$','$$']], processEscapes: true},
  options: {skipHtmlTags: ['script','noscript','style','textarea','pre','code']},
  chtml: {scale: 1.0}
};
</script>
<script src="mathjax.js"></script>
</head><body>
<article>
<header>
<h1>@@TITLE@@</h1>
<div class="author">@@AUTHOR@@</div>
</header>
@@BODY@@
<h2 class="bib-h">References</h2>
<ul class="bib">@@BIB@@</ul>
</article>
</body></html>
"""


def strip_underbrace(s):
    """Flatten \\underbrace{X}_{Y} to X (MathJax renders it unreadably small)."""
    key = "\\underbrace{"
    out, i = [], 0
    while True:
        j = s.find(key, i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        k = j + len(key)
        depth, start = 1, k
        while k < len(s) and depth:
            if s[k] == '{':
                depth += 1
            elif s[k] == '}':
                depth -= 1
            k += 1
        out.append(s[start:k - 1])
        i = k
        if i < len(s) and s.startswith("_{", i):
            depth, m = 1, i + 2
            while m < len(s) and depth:
                if s[m] == '{':
                    depth += 1
                elif s[m] == '}':
                    depth -= 1
                m += 1
            i = m
    return "".join(out)


_ACCENTS = {'"': "", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303"}


def clean_math(s):
    s = strip_underbrace(s)
    s = s.replace("\\!", "")      # this build's MathJax renders \! as inverted question mark
    s = s.replace("\\ell", "l")
    # glyphs missing from this MathJax build -> safe equivalents
    s = s.replace("\\not\\ni", "\\neq")
    s = s.replace("\\ni", "\\in")
    s = s.replace("\\|", "\\lVert ")     # \|e\|_2 was rendering as boxes
    s = s.replace("\\lVert e\\lVert", "\\lVert e\\rVert")
    # unicode fallbacks for glyphs this MathJax build lacks
    s = s.replace("\\pm", "\u00b1")
    s = s.replace("\\mp", "\u2213")
    s = s.replace("\\times", "\u00d7")
    s = s.replace("\\approx", "\u2248")
    s = s.replace("\\to", "\u2192")
    s = s.replace("\\Delta", "\u0394")
    s = s.replace("\\sigma", "\u03c3")
    s = s.replace("\\sim", "\u223c")
    s = s.replace("\\textbf", "\\mathbf")
    # \texttt -> \mathtt first, then pick a tight upright form for identifiers
    s = s.replace("\\texttt", "\\mathtt")
    for ident in ("carried", "static", "reset", "shuffled", "concat"):
        s = s.replace("\\mathtt{" + ident + "}", "\\mathrm{" + ident + "}")
    return s


def _brace_arg(text, cmd):
    """Return the argument of `cmd{...}` with balanced braces, as a fake match."""
    i = text.find(cmd)
    if i < 0:
        return None
    j = text.find('{', i + len(cmd))
    if j < 0:
        return None
    depth, k = 1, j + 1
    while k < len(text) and depth:
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
        k += 1
    val = text[j + 1:k - 1]

    class _M:
        def group(self, _n=1):
            return val
    return _M()


def _tabular(text):
    """Return the rows of the first tabular, tolerating nested braces in the spec."""
    i = text.find('\\begin{tabular}')
    if i < 0:
        return None
    # skip the column-spec argument, which may itself contain @{...}
    j = text.find('{', i)
    depth, k = 1, j + 1
    while k < len(text) and depth:
        if text[k] == '{':
            depth += 1
        elif text[k] == '}':
            depth -= 1
        k += 1
    end = text.find('\\end{tabular}', k)
    val = text[k:end]

    class _M:
        def group(self, _n=1):
            return val
    return _M()


def main():
    src = io.open(TEX, encoding="utf-8").read()
    pre, body = src.split("\\begin{document}", 1)

    # expand \newcommand macros so MathJax only sees plain TeX
    macros = dict(re.findall(
        r"\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}", pre))
    for name, val in macros.items():
        repl = re.sub(r"\\\\", r"\\", val)
        pat = re.compile(r"\\" + name + r"(?![A-Za-z])")
        pre = pat.sub(lambda _m, b=repl: b, pre)
        body = pat.sub(lambda _m, b=repl: b, body)

    if "\\begin{thebibliography}" in body:
        body, bib_part = body.split("\\begin{thebibliography}", 1)
    else:
        bib_part = ""

    title = re.search(r"\\title\{(.*?)\}\s*\n", pre, re.S).group(1)
    title = re.sub(r"\\\\\s*", " ", title).replace("\\vspace{-1.2em}", "").strip()
        # author block: first line is the name, remaining \\small lines are affiliation
    amatch = re.search(r"\\author\{((?:[^{}]|\{[^{}]*\})*)\}\s*\n", pre, re.S)
    author_html = ""
    if amatch:
        parts = [p.strip() for p in amatch.group(1).split(chr(92) * 2) if p.strip()]
        name = parts[0].replace(chr(92), "").strip()
        affil = []
        for extra in parts[1:]:
            extra = extra.replace("\\small", "").strip()
            extra = re.sub(r"\\texttt\{([^}]*)\}", r"\1", extra)
            extra = re.sub(r"\\", "", extra).strip()
            if extra:
                affil.append(extra)
        author_html = f'<div class="author-name">{name}</div>'
        if affil:
            author_html += '<div class="affil">' + " · ".join(affil) + "</div>"

    # ---- numbering, single pass over each construct type
    ref_map = {}
    fig_nums, tab_nums = [], []
    for m in re.finditer(r"\\begin\{(figure|table)\}(.*?)\\end\{\1\}", body, re.S):
        chunk = m.group(2)
        lab = re.search(r"\\label\{([^}]+)\}", chunk)
        if m.group(1) == "figure":
            fig_nums.append(m.group(0))
            if lab:
                ref_map[lab.group(1)] = str(len(fig_nums))
        else:
            tab_nums.append(m.group(0))
            if lab:
                ref_map[lab.group(1)] = str(len(tab_nums))

    eq_blocks = re.findall(r"\\begin\{equation\}.*?\\end\{equation\}", body, re.S)
    for i, b in enumerate(eq_blocks, 1):
        lab = re.search(r"\\label\{([^}]+)\}", b)
        if lab:
            ref_map[lab.group(1)] = f"({i})"

    # section labels
    sec_no = sub_no = 0
    for m in re.finditer(r"\\(section|subsection)\{([^}]*)\}", body):
        if m.group(1) == "section":
            sec_no += 1
            sub_no = 0
            num = str(sec_no)
        else:
            sub_no += 1
            num = f"{sec_no}.{sub_no}"
        lab = re.search(r"\\label\{([^}]+)\}", body[m.end():m.end() + 100])
        if lab and lab.group(1) not in ref_map:
            ref_map[lab.group(1)] = num

    # ---- inline formatting
    def inline(text):
        maths = []

        def stash(m):
            maths.append(clean_math(m.group(0)))
            return f"\x00{len(maths)-1}\x00"

        text = re.sub(r"\$\$(.+?)\$\$", stash, text, flags=re.S)
        text = re.sub(r"\$(.+?)\$", stash, text, flags=re.S)

        text = text.replace("\\%", "%").replace("\\_", "_").replace("\\&", "&")
        text = text.replace("\\,", " ").replace("\\ ", " ").replace("~", " ")
        text = re.sub(r"\\label\{[^}]+\}", "", text)
        text = re.sub(r"\\paragraph\{([^{}]*)\}", r'<span class="run-in">\1</span> ', text)
        text = re.sub(r"\\textbf\{([^{}]*)\}", r"<strong>\1</strong>", text)
        text = re.sub(r"\\emph\{([^{}]*)\}", r"<em>\1</em>", text)
        text = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", text)
        text = text.replace("{}", "")
        text = re.sub(r"\\url\{([^}]*)\}", r'<a href="\1">\1</a>', text)

        def _accent(m):
            table = {
                ('"', 'u'): "\u00fc", ('"', 'a'): "\u00e4", ('"', 'o'): "\u00f6",
                ("'", 'e'): "\u00e9", ("'", 'a'): "\u00e1", ("'", 'i'): "\u00ed",
                ("'", 'o'): "\u00f3", ("'", 'u'): "\u00fa",
                ('`', 'a'): "\u00e0", ('`', 'e'): "\u00e8",
                ('^', 'a'): "\u00e2", ('^', 'e'): "\u00ea", ('^', 'o'): "\u00f4",
                ('~', 'n'): "\u00f1", ('~', 'a'): "\u00e3",
            }
            return table.get((m.group(1), m.group(2)), m.group(2))

        text = re.sub(r"""\{\\(["'`^~])\{?([A-Za-z])\}?\}""", _accent, text)

        def ref(m):
            return ref_map.get(m.group(1), "?")

        text = re.sub(r"\\eqref\{([^}]+)\}", ref, text)
        text = re.sub(r"\\S\\ref\{([^}]+)\}", lambda m: "&sect;" + ref(m), text)
        text = re.sub(r"\\S([0-9.]+)", lambda m: "&sect;" + m.group(1), text)
        text = re.sub(r"\\ref\{([^}]+)\}", ref, text)

        text = re.sub(
            r"\\cite[pt]?\{([^}]+)\}",
            lambda m: '<span class="cite">[' + ", ".join(
                k.strip() for k in m.group(1).split(",")) + "]</span>",
            text)

        text = re.sub(r"\x00(\d+)\x00", lambda m: maths[int(m.group(1))], text)
        return text.replace("---", "&mdash;").replace("--", "&ndash;")

    # ---- block pass
    out = []
    labels = ("figure", "table", "equation")
    pat = re.compile(
        r"\\begin\{(figure|table|equation)\}.*?\\end\{\1\}", re.S)

    seq = []
    pos = 0
    for m in pat.finditer(body):
        if m.start() > pos:
            seq.append(("text", body[pos:m.start()]))
        seq.append((m.group(1), m.group(0)))
        pos = m.end()
    seq.append(("text", body[pos:]))

    eq_seen = 0
    s_no = sub_no_v = 0
    for kind, raw in seq:
        if kind == "text":
            seg = raw
            abstract = None
            am = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", seg, re.S)
            if am:
                abstract = re.sub(r"\s+", " ", am.group(1)).strip()
                seg = seg[:am.start()] + seg[am.end():]

            def render_list(m):
                items = re.findall(r"\\item\s+(.*?)(?=\\item|\\end\{enumerate\})",
                                   m.group(0), re.S)
                return "<ol>" + "".join(
                    f"<li>{inline(i.strip())}</li>" for i in items) + "</ol>"

            seg = re.sub(r"\\begin\{enumerate\}.*?\\end\{enumerate\}", render_list,
                         seg, flags=re.S)
            seg = re.sub(r"\\(begin|end)\{(itemize|center|small)\}", "", seg)
            seg = seg.replace("\\maketitle", "")

            if abstract:
                out.append('<div class="abstract"><span class="abstract-label">'
                           'Abstract.</span> ' + inline(abstract) + "</div>")

            while True:
                hm = re.search(r"\\(section|subsection)\{([^}]*)\}", seg)
                if not hm:
                    break
                for p in [x.strip() for x in re.split(r"\n\s*\n", seg[:hm.start()]) if x.strip()]:
                    out.append(f"<p>{inline(p)}</p>")
                if hm.group(1) == "section":
                    s_no += 1
                    sub_no_v = 0
                    out.append(f'<h2>{s_no}. {inline(hm.group(2))}</h2>')
                else:
                    sub_no_v += 1
                    out.append(f"<h3>{s_no}.{sub_no_v} {inline(hm.group(2))}</h3>")
                seg = seg[hm.end():]
            for p in [x.strip() for x in re.split(r"\n\s*\n", seg) if x.strip()]:
                out.append(f"<p>{inline(p)}</p>")

        elif kind == "figure":
            num = fig_nums.index(raw) + 1
            cap = _brace_arg(raw, "\\caption")
            inc = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", raw)
            src = inc.group(1).replace(".pdf", ".png") if inc else ""
            out.append(
                f'<figure id="fig{num}"><img src="{src}" alt="Figure {num}">'
                f'<figcaption><span class="fignum">Figure {num}.</span> '
                f'{inline(cap.group(1).strip()) if cap else ""}</figcaption></figure>')

        elif kind == "table":
            num = tab_nums.index(raw) + 1
            cap = _brace_arg(raw, "\\caption")
            tab = _tabular(raw)
            rows = ""
            if tab:
                cleaned = re.sub(r"\\(top|mid|bottom)rule", "", tab.group(1))
                # the column spec can leak in as a leading {@{}lll@{}} group
                cleaned = re.sub(r"^\s*\{@\{\}[A-Za-z@{}| ]+\}\s*", "", cleaned)
                cleaned = cleaned.lstrip()
                trs = []
                for r in [x.strip() for x in cleaned.split("\\\\") if x.strip()]:
                    cells = [c.strip() for c in r.split("&")]
                    trs.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                rows = "\n".join(trs)
            out.append(
                f'<table id="tab{num}"><caption><span class="fignum">Table {num}.</span> '
                f'{inline(cap.group(1).strip()) if cap else ""}</caption>'
                f"<tbody>{rows}</tbody></table>")

        elif kind == "equation":
            eq_seen += 1
            inner = raw.replace("\\begin{equation}", "").replace("\\end{equation}", "")
            inner = re.sub(r"\\label\{[^}]+\}", "", inner).strip()
            out.append(
                f'<div class="eq" id="eq{eq_seen}"><div class="eq-body">'
                f"$${clean_math(inner)}$$</div>"
                f'<div class="eq-num">({eq_seen})</div></div>')

    bib = []
    for m in re.finditer(
            r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\}|$)",
            bib_part, re.S):
        key, txt = m.group(1), m.group(2).strip()
        txt = re.sub(r"\\newblock\s*", "<br>", txt)
        txt = re.sub(r"\s+", " ", txt)
        bib.append(f'<li id="ref-{key}"><span class="cite">[{key}]</span> {inline(txt)}</li>')

    doc = (TEMPLATE.replace("@@TITLE@@", inline(title))
                   .replace("@@AUTHOR@@", author_html)
                   .replace("@@BODY@@", "\n".join(out))
                   .replace("@@BIB@@", "".join(bib)))
    io.open(OUT, "w", encoding="utf-8").write(doc)
    print("wrote", OUT, len(doc), "bytes")


if __name__ == "__main__":
    main()
