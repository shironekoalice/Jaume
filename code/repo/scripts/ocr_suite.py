"""OCR-oriented task family, rendered locally for a text-reading VLM probe.

Three sub-tasks, each isolating a different OCR competence:

  doc_field    read a labelled field out of an invoice page (total, tax, ...)
  char_read    read a short alphanumeric string rendered in a grid
  text_locate  pick which of four labelled boxes contains a given string

Rationale: the basic-visual suite saturated on shape/colour/spatial tasks but
never tested reading. OCR is the capability that most directly stresses
resolution and glyph fidelity, so it is a separate family rather than another
row in the same table.

All rendering is local (Pillow + DejaVu); no download dependency.
"""
import random
import string
from PIL import Image, ImageDraw, ImageFont

W, H = 320, 400
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

VENDORS = ["Acme", "Northwind", "Globex", "Initech", "Umbrella", "Cyberdyne"]
ITEMS = ["Widget", "Gasket", "Bracket", "Cable", "Module", "Valve"]


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.truetype(FONT, size)


# ------------------------------------------------------------------ doc_field
def gen_doc_field(rng):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    vendor = rng.choice(VENDORS)
    inv = rng.randint(100000, 999999)
    d.text((18, 14), vendor, font=_font(FONT_B, 24), fill="black")
    d.text((18, 46), "INVOICE", font=_font(FONT_B, 16), fill="black")
    d.text((18, 72), "No. %d" % inv, font=_font(FONT, 14), fill="black")

    top = 108
    d.rectangle([14, top, W - 14, top + 22], fill="#1e1e1e")
    d.text((22, top + 4), "Item", font=_font(FONT_B, 13), fill="white")
    d.text((190, top + 4), "Qty", font=_font(FONT_B, 13), fill="white")
    d.text((250, top + 4), "Amt", font=_font(FONT_B, 13), fill="white")

    n = rng.randint(3, 5)
    total = 0
    y = top + 28
    for _ in range(n):
        item = rng.choice(ITEMS)
        qty = rng.randint(1, 9)
        price = rng.choice([5, 10, 15, 20, 25, 40, 50, 75])
        amt = qty * price
        total += amt
        d.text((22, y), item, font=_font(FONT, 13), fill="black")
        d.text((196, y), str(qty), font=_font(FONT, 13), fill="black")
        d.text((252, y), "$%d" % amt, font=_font(FONT, 13), fill="black")
        d.line([14, y + 20, W - 14, y + 20], fill="#d0d0d0")
        y += 26

    tax = int(round(total * 0.08))
    grand = total + tax
    y += 12
    d.text((170, y), "Subtotal", font=_font(FONT, 13), fill="black")
    d.text((256, y), "$%d" % total, font=_font(FONT, 13), fill="black")
    d.text((170, y + 22), "Tax", font=_font(FONT, 13), fill="black")
    d.text((256, y + 22), "$%d" % tax, font=_font(FONT, 13), fill="black")
    d.rectangle([164, y + 46, W - 14, y + 70], fill="#ececec")
    d.text((170, y + 51), "TOTAL", font=_font(FONT_B, 14), fill="black")
    d.text((250, y + 51), "$%d" % grand, font=_font(FONT_B, 14), fill="black")

    which = rng.choice(["total", "invoice", "subtotal"])
    ans = {"total": str(grand), "invoice": str(inv),
           "subtotal": str(total)}[which]
    # answers are wide-range integers, so options are not a fixed small set;
    # we score exact string match instead of multiple choice.
    q = {"total": "What is the total amount?",
         "invoice": "What is the invoice number?",
         "subtotal": "What is the subtotal?"}[which]
    return img, "doc_field", which, ans, "free", q


# ------------------------------------------------------------------ char_read
ALPHA = string.ascii_uppercase
DIGITS = string.digits


def gen_char_read(rng):
    """A 4-character code rendered in large monospace, in a random style."""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    alphabet = rng.choice([ALPHA, DIGITS, ALPHA + DIGITS])
    code = "".join(rng.choice(alphabet) for _ in range(4))
    style = rng.choice(["plain", "inverted", "bordered"])
    if style == "plain":
        d.text((48, 150), code, font=_font(FONT_M, 64), fill="black")
    elif style == "inverted":
        d.rectangle([20, 120, W - 20, 250], fill="black")
        d.text((48, 150), code, font=_font(FONT_M, 64), fill="white")
    else:
        d.rectangle([20, 120, W - 20, 250], outline="black", width=3)
        d.text((48, 150), code, font=_font(FONT_M, 64), fill="black")
    # light noise: a few thin rules, so the model cannot rely on a blank page
    for _ in range(rng.randint(0, 3)):
        yy = rng.randint(20, H - 20)
        d.line([10, yy, W - 10, yy], fill="#e0e0e0", width=1)
    sub = {"plain": "plain", "inverted": "inverted", "bordered": "bordered"}
    return img, "char_read", sub[style], code, "free", "What is the code?"


# ---------------------------------------------------------------- text_locate
WORDS = ["ALPHA", "BRAVO", "DELTA", "OMEGA", "SIGMA", "THETA", "GAMMA"]


def gen_text_locate(rng):
    """Four labelled boxes; two share a word family to force careful reading."""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    words = rng.sample(WORDS, 4)
    target = rng.choice(words)
    boxes = [(20, 40), (170, 40), (20, 230), (170, 230)]
    for i, (bx, by) in enumerate(boxes):
        d.rectangle([bx, by, bx + 130, by + 150], outline="black", width=2)
        d.text((bx + 30, by + 60), words[i], font=_font(FONT_B, 20), fill="black")
        d.text((bx + 8, by + 8), "box %d" % (i + 1), font=_font(FONT, 12),
               fill="#666666")
    idx = words.index(target)
    ans = "box %d" % (idx + 1)
    return img, "text_locate", "which_box", ans, "free",         "Which box contains the word %s?" % target


GENERATORS = [gen_doc_field, gen_char_read, gen_text_locate]
OCR_FAMILIES = ["doc_field", "char_read", "text_locate"]


def build(n, seed):
    out = []
    for i in range(n):
        rng = random.Random(seed * 1000003 + i)
        g = GENERATORS[i % len(GENERATORS)]
        img, fam, which, ans, opts, q = g(rng)
        out.append((img, fam, which, ans, opts, q))
    return out
