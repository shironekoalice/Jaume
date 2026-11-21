# -*- coding: utf-8 -*-
"""Render synthetic *but realistic* document images for VLM document QA.

Why this task
-------------
The synthetic shape-counting probe proved the visual bridge works but has no
product value. Document understanding (invoices, tables, forms) is the core
commercial VLM use case and exercises exactly what shape counting does not:
reading small text, aligning spatial layout, and answering a structured query
about a field that must be *located* in the image.

Everything is rendered locally with Pillow, so there is no download
dependency and the label distribution is fully controlled.
"""
import random
from PIL import Image, ImageDraw, ImageFont

W, H = 384, 480          # document canvas
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

VENDORS = ["Acme Corp", "Northwind", "Globex", "Initech", "Umbrella",
           "Stark Industries", "Wayne Ent", "Cyberdyne"]
ITEMS = ["Widget", "Gasket", "Bracket", "Cable", "Module", "Valve",
         "Sensor", "Adapter", "Housing", "Bearing"]


def _font(size, bold=False):
    path = FONT_CANDIDATES[1] if bold else FONT_CANDIDATES[0]
    return ImageFont.truetype(path, size)


def make_invoice(seed):
    """Render one invoice; return (PIL image, dict of ground-truth fields)."""
    rng = random.Random(seed)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    vendor = rng.choice(VENDORS)
    inv_no = rng.randint(10000, 99999)
    date = "2026-%02d-%02d" % (rng.randint(1, 12), rng.randint(1, 28))

    d.text((20, 18), vendor, font=_font(26, bold=True), fill="black")
    d.text((20, 54), "INVOICE", font=_font(18, bold=True), fill="black")
    d.text((20, 82), "No. %d" % inv_no, font=_font(15), fill="black")
    d.text((20, 104), "Date: %s" % date, font=_font(15), fill="black")
    d.text((20, 126), "Bill To: %s" % rng.choice(VENDORS),
           font=_font(14), fill="black")

    # table header
    top = 168
    d.rectangle([16, top, W - 16, top + 26], fill="#222")
    d.text((26, top + 5), "Item", font=_font(14, bold=True), fill="white")
    d.text((210, top + 5), "Qty", font=_font(14, bold=True), fill="white")
    d.text((270, top + 5), "Price", font=_font(14, bold=True), fill="white")
    d.text((332, top + 5), "Amt", font=_font(14, bold=True), fill="white")

    n_rows = rng.randint(3, 6)
    rows = []
    total = 0
    y = top + 32
    for _ in range(n_rows):
        item = rng.choice(ITEMS)
        qty = rng.randint(1, 9)
        price = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 75, 100])
        amt = qty * price
        total += amt
        rows.append((item, qty, price, amt))
        d.text((26, y), item, font=_font(14), fill="black")
        d.text((214, y), str(qty), font=_font(14), fill="black")
        d.text((266, y), "$%d" % price, font=_font(14), fill="black")
        d.text((332, y), "$%d" % amt, font=_font(14), fill="black")
        d.line([16, y + 24, W - 16, y + 24], fill="#ccc")
        y += 30

    # totals block
    y += 14
    tax = int(round(total * 0.08))
    grand = total + tax
    d.text((210, y), "Subtotal", font=_font(14), fill="black")
    d.text((330, y), "$%d" % total, font=_font(14), fill="black")
    d.text((210, y + 24), "Tax 8%", font=_font(14), fill="black")
    d.text((330, y + 24), "$%d" % tax, font=_font(14), fill="black")
    d.rectangle([204, y + 50, W - 16, y + 78], fill="#eee")
    d.text((210, y + 56), "TOTAL", font=_font(16, bold=True), fill="black")
    d.text((322, y + 56), "$%d" % grand, font=_font(16, bold=True), fill="black")

    gt = {
        "vendor": vendor,
        "invoice": inv_no,
        "date": date,
        "n_rows": n_rows,
        "subtotal": total,
        "tax": tax,
        "total": grand,
        "first_item": rows[0][0],
        "first_qty": rows[0][1],
    }
    return img, gt


# Questions must be answerable from pixels but NOT be a single global pool:
# the answer depends on the rendered content of this specific image.
QUESTION_KINDS = [
    ("total", "What is the grand total?", 0),
    ("subtotal", "What is the subtotal?", 1),
    ("invoice", "What is the invoice number?", 2),
    ("n_rows", "How many line items are there?", 3),
    ("first_qty", "What is the quantity of the first line item?", 4),
]


def build(n, seed):
    """Return list of (PIL image, question_kind, answer_string)."""
    out = []
    for i in range(n):
        img, gt = make_invoice(seed * 1000003 + i)
        kind, _q, _ = QUESTION_KINDS[i % len(QUESTION_KINDS)]
        ans = str(gt[kind])
        out.append((img, kind, ans))
    return out


def make_question(kind):
    for k, q, _ in QUESTION_KINDS:
        if k == kind:
            return q
    raise KeyError(kind)
