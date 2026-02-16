import random
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("../../data/timer_chars/train")  # or val
CLASSES = [str(i) for i in range(10)] + ["colon", "dot"]

def main():
    for cls in CLASSES:
        imgs = list((ROOT / cls).glob("*.png"))
        print(cls, len(imgs))
        if not imgs:
            continue

        picks = random.sample(imgs, min(40, len(imgs)))
        tiles = []
        for p in picks:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_NEAREST)
            tiles.append(im)

        if not tiles:
            continue

        # make a 5x8 grid
        grid_h, grid_w = 5, 8
        tiles = tiles[:grid_h * grid_w]
        while len(tiles) < grid_h * grid_w:
            tiles.append(np.zeros((32,32), dtype=np.uint8))

        rows = []
        for r in range(grid_h):
            rows.append(np.hstack(tiles[r*grid_w:(r+1)*grid_w]))
        grid = np.vstack(rows)

        out = Path("data/timer_chars/sanity")
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{cls}.png"), grid)
    print("Wrote grids to data/timer_chars/sanity/*.png")

if __name__ == "__main__":
    main()
