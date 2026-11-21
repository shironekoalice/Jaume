import sys
sys.path.insert(0, "/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo/scripts")
from ocr_suite import build

def split_tiles(img, grid, overlap=0.15):
    W, H = img.size
    tw, th = W / grid, H / grid
    ow, oh = tw * overlap, th * overlap
    tiles = []
    for gy in range(grid):
        for gx in range(grid):
            x0 = max(0, int(gx * tw - ow)); y0 = max(0, int(gy * th - oh))
            x1 = min(W, int((gx + 1) * tw + ow)); y1 = min(H, int((gy + 1) * th + oh))
            tiles.append(img.crop((x0, y0, x1, y1)))
    return tiles

items = [b for b in build(200, seed=31) if b[1] == "text_locate"]
img = items[0][0]
print("page size:", img.size)
for g in (1, 2, 3):
    ts = split_tiles(img, g)
    print("grid", g, "tiles", len(ts), "each", ts[0].size)
    ts[0].save("/mnt/workspace/output/tile_g" + str(g) + ".png")
