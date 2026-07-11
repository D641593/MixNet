# -*- coding: utf-8 -*-
"""
apply_compat_patches.py
-------------------------
One-time compatibility patch for running MixNet training on a modern
environment (NumPy >= 1.24, Shapely >= 2.0, recent PyTorch). The repo was
written against older versions of these libraries; without these patches
training crashes with version-mismatch errors that have nothing to do with
your dataset.

Run once, from the MixNet repo root:

    python3 apply_compat_patches.py

Safe to re-run: every patch checks whether it's already applied before
touching a file.

What it does
------------
1. Replaces removed NumPy 1.24+ aliases (`np.float`, `np.int`, `np.bool`,
   `np.int0`) across dataset/, util/, network/ with their modern
   equivalents (`float`, `int`, `bool`, `np.intp`).
2. Fixes `dataset/dataload_midline.py`'s `get_more_points()`, which relied
   on `np.array(<shapely geometry>)` -- this implicit conversion was
   removed in Shapely >= 2.0. Patched to pull coordinates out via
   `.geoms` explicitly.
3. Fixes `network/Reg_loss.py`'s `np.reshape(..., newshape=...)` call --
   the `newshape` keyword was renamed to `shape` in newer NumPy.
4. Adds `config.pos = False` and `config.resume = None` defaults to
   `cfglib/config.py`. `network/textnet.py` reads both, but neither is
   ever exposed as a CLI flag or given a default upstream, so building a
   model (train OR eval OR standalone inference) raises AttributeError
   without this.
5. Removes the `mmcv` dependency from `dataset/TD500HUST_Text.py` and
   `dataset/TD500HUST_Text_mid.py`, replacing its one used function
   (`mmcv.list_from_file`) with a 3-line local equivalent. `mmcv`'s build
   is heavy and fragile on newer environments (commonly fails with
   `ModuleNotFoundError: No module named 'pkg_resources'` during
   `pip install mmcv`, among other issues) -- yet the only thing it's
   used for here is reading a text file line by line, and `dataset`'s
   `__init__.py` imports these files unconditionally just to expose the
   TD500HUST loader class, so everyone paid for that dependency even if
   they never train on TD500HUST.

What it does NOT do (separate, non-code prerequisites)
--------------------------------------------------------
- Install missing pip packages. Importing `dataset` (even just to use one
  loader) eagerly imports every dataset loader module, which pulls in
  extra dependencies not listed in requirements.txt:
      pip install pycocotools ujson pytorch-ssim --break-system-packages
  (mmcv is deliberately NOT in this list -- this script removes the need
  for it, see patch 5 below.)
- Provide the ImageNet/Synth-pretrained backbone weight. Training from
  scratch (no --resume) makes network/layers/FSNet.py try to auto-load
  ./pretrained/triHRnet_Synth_weight.pth, which isn't shipped with the
  repo and isn't linked anywhere in the README. Always pass --resume
  pointing at one of the released MixNet checkpoints (e.g.
  model/Totaltext_mid/MixNet_FSNet_M_622.pth) when starting training --
  this both avoids the crash and gives you a much better initialization
  than training from scratch anyway.
"""
import os
import re

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def patch_numpy_aliases():
    targets = []
    for base in ("dataset", "util", "network"):
        base_path = os.path.join(REPO_ROOT, base)
        if not os.path.isdir(base_path):
            continue
        for root, _, files in os.walk(base_path):
            for fname in files:
                if fname.endswith(".py"):
                    targets.append(os.path.join(root, fname))

    pattern_int0 = re.compile(r"\bnp\.int0\b")
    pattern_float = re.compile(r"\bnp\.float\b")
    pattern_bool = re.compile(r"\bnp\.bool\b")
    pattern_int = re.compile(r"\bnp\.int\b")

    n_changed = 0
    for path in targets:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        new_src = pattern_int0.sub("np.intp", src)
        new_src = pattern_float.sub("float", new_src)
        new_src = pattern_bool.sub("bool", new_src)
        new_src = pattern_int.sub("int", new_src)
        if new_src != src:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
            n_changed += 1
            print(f"[numpy-aliases] patched {os.path.relpath(path, REPO_ROOT)}")
    print(f"[numpy-aliases] {n_changed} file(s) changed")


def patch_shapely_midline():
    path = os.path.join(REPO_ROOT, "dataset", "dataload_midline.py")
    if not os.path.exists(path):
        print("[shapely] dataset/dataload_midline.py not found, skipping")
        return

    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    if "hasattr(multipoint, \"geoms\")" in src or "hasattr(multipoint, 'geoms')" in src:
        print("[shapely] already patched")
        return

    old = (
        "    def get_more_points(self, midline, target_pts):\n"
        "        line = LineString(midline)\n"
        "        distances = np.linspace(0, line.length, target_pts)\n"
        "        points = [line.interpolate(distance) for distance in distances]\n"
        "        multipoint = unary_union(points) \n"
        "        midline = np.array(multipoint)\n"
        "        return midline\n"
    )
    new = (
        "    def get_more_points(self, midline, target_pts):\n"
        "        line = LineString(midline)\n"
        "        distances = np.linspace(0, line.length, target_pts)\n"
        "        points = [line.interpolate(distance) for distance in distances]\n"
        "        multipoint = unary_union(points)\n"
        "        # Shapely >= 2.0: geometries no longer support np.array(geom)\n"
        "        # directly; pull coordinates out explicitly instead.\n"
        "        if hasattr(multipoint, \"geoms\"):\n"
        "            midline = np.array([[p.x, p.y] for p in multipoint.geoms])\n"
        "        else:\n"
        "            midline = np.array([[multipoint.x, multipoint.y]])\n"
        "        return midline\n"
    )

    if old not in src:
        print("[shapely] WARNING: expected get_more_points() body not found verbatim; "
              "skipping (file may already differ from the version this patch targets). "
              "Patch dataset/dataload_midline.py::get_more_points manually -- see this "
              "script's docstring for the intended fix.")
        return

    src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("[shapely] patched dataset/dataload_midline.py::get_more_points")


def patch_reshape_kwarg():
    path = os.path.join(REPO_ROOT, "network", "Reg_loss.py")
    if not os.path.exists(path):
        print("[reshape] network/Reg_loss.py not found, skipping")
        return
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    old = "np.reshape(pidxall, newshape=(batch_size, -1))"
    new = "np.reshape(pidxall, (batch_size, -1))"
    if old not in src:
        if new in src:
            print("[reshape] already patched")
        else:
            print("[reshape] WARNING: expected np.reshape(...) call not found, skipping")
        return
    src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("[reshape] patched network/Reg_loss.py")


def patch_cfg_defaults():
    path = os.path.join(REPO_ROOT, "cfglib", "config.py")
    if not os.path.exists(path):
        print("[cfg-defaults] cfglib/config.py not found, skipping")
        return
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if "config.pos = False" in src and "config.resume = None" in src:
        print("[cfg-defaults] already patched")
        return

    marker = "def update_config(config, extra_config):"
    if marker not in src:
        print("[cfg-defaults] WARNING: could not find insertion point, skipping")
        return

    addition = (
        "# not exposed via CLI upstream, but network/textnet.py reads them --\n"
        "# must default to something or model construction raises AttributeError\n"
        "config.pos = False\n"
        "config.resume = None\n\n\n"
    )
    src = src.replace(marker, addition + marker)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("[cfg-defaults] patched cfglib/config.py")


def patch_remove_mmcv_dependency():
    """`mmcv` is only ever used for one helper (`mmcv.list_from_file`, which
    just reads a text file into a list of stripped lines) inside
    dataset/TD500HUST_Text.py and dataset/TD500HUST_Text_mid.py -- but
    importing `dataset` at all (even to use an unrelated loader) pulls
    those files in, so everyone pays for the full mmcv install. mmcv's
    build can fail on newer environments (e.g. "ModuleNotFoundError: No
    module named 'pkg_resources'" during pip install, or missing build
    tools) even though the actual functionality used is three lines of
    Python. Replace the import with a tiny local equivalent instead.
    """
    targets = [
        os.path.join(REPO_ROOT, "dataset", "TD500HUST_Text.py"),
        os.path.join(REPO_ROOT, "dataset", "TD500HUST_Text_mid.py"),
    ]
    helper = (
        "\n\ndef _list_from_file(path):\n"
        "    # tiny stand-in for mmcv.list_from_file -- avoids depending on\n"
        "    # mmcv just for this one helper (see apply_compat_patches.py)\n"
        "    with open(path, encoding=\"utf-8-sig\") as f:\n"
        "        return [line.rstrip(\"\\n\").rstrip(\"\\r\") for line in f]\n"
    )
    for path in targets:
        if not os.path.exists(path):
            print(f"[mmcv] {os.path.relpath(path, REPO_ROOT)} not found, skipping")
            continue
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        if "_list_from_file" in src:
            print(f"[mmcv] {os.path.relpath(path, REPO_ROOT)} already patched")
            continue
        if "import mmcv" not in src:
            print(f"[mmcv] WARNING: 'import mmcv' not found in "
                  f"{os.path.relpath(path, REPO_ROOT)}, skipping")
            continue
        src = src.replace("import mmcv\n", "", 1)
        src = src.replace("mmcv.list_from_file(", "_list_from_file(")
        # insert the helper right after the import block (before the class)
        marker = "\nclass "
        idx = src.find(marker)
        if idx == -1:
            print(f"[mmcv] WARNING: could not find insertion point in "
                  f"{os.path.relpath(path, REPO_ROOT)}, skipping")
            continue
        src = src[:idx] + helper + src[idx:]
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"[mmcv] patched {os.path.relpath(path, REPO_ROOT)} (mmcv dependency removed)")


def main():
    patch_numpy_aliases()
    patch_shapely_midline()
    patch_reshape_kwarg()
    patch_cfg_defaults()
    patch_remove_mmcv_dependency()
    print("\nDone. Remember the two non-code prerequisites in this script's docstring:")
    print("  pip install pycocotools ujson pytorch-ssim --break-system-packages")
    print("  always pass --resume <a released .pth checkpoint> when training")


if __name__ == "__main__":
    main()