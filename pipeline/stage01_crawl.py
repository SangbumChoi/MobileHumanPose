"""Stage 1 - Crawl images.

Robust, multi-source image acquisition that always produces *something* so the
rest of the pipeline can run regardless of the sandbox network policy:

  1. Wikimedia Commons API  (license-clean, reliable, default)
  2. DuckDuckGo image search (best-effort, opportunistic)
  3. Bundled fallback assets (pipeline/assets/fallback) committed to the repo

Output: ``work/01_raw/`` with the downloaded JPEGs and a ``manifest.json``.
"""

import argparse
import io
import os
import os.path as osp
import time

import requests
from PIL import Image

from common import (RAW_DIR, FALLBACK_DIR, ensure_dir, save_json, get_logger)

log = get_logger("crawl")
UA = "MobileHumanPose-Pipeline/1.0 (research; contact: sbchoi@toss.im)"
HEADERS = {"User-Agent": UA}


def _save_image(content, dst_path, min_side=160):
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return False
    if min(img.size) < min_side:
        return False
    img.save(dst_path, "JPEG", quality=92)
    return True


def from_wikimedia(query, limit):
    """Search Wikimedia Commons for files matching the query."""
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": limit * 3,
        "prop": "imageinfo", "iiprop": "url|mime|extmetadata", "iiurlwidth": 640,
    }
    out = []
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
    except Exception as e:
        log.info("Wikimedia search failed: %s", e)
        return out
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        mime = info.get("mime", "")
        if not mime.startswith("image/"):
            continue
        src = info.get("thumburl") or info.get("url")
        if src:
            lic = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "?")
            out.append({"url": src, "source": "wikimedia", "license": lic,
                        "title": page.get("title", "")})
    return out


def from_duckduckgo(query, limit):
    """Best-effort DuckDuckGo image search (no API key)."""
    out = []
    try:
        s = requests.Session()
        s.headers.update(HEADERS)
        token = s.get("https://duckduckgo.com/", params={"q": query}, timeout=15).text
        import re
        m = re.search(r'vqd=["\']?([\d-]+)', token)
        if not m:
            return out
        res = s.get("https://duckduckgo.com/i.js",
                    params={"l": "us-en", "o": "json", "q": query,
                            "vqd": m.group(1), "f": "", "p": "1"},
                    headers={**HEADERS, "Referer": "https://duckduckgo.com/"}, timeout=15)
        for item in res.json().get("results", [])[:limit * 2]:
            out.append({"url": item.get("image"), "source": "duckduckgo", "license": "?",
                        "title": item.get("title", "")})
    except Exception as e:
        log.info("DuckDuckGo search failed: %s", e)
    return out


def use_fallback(limit):
    out = []
    if osp.isdir(FALLBACK_DIR):
        for fn in sorted(os.listdir(FALLBACK_DIR)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append({"url": "file://" + osp.join(FALLBACK_DIR, fn),
                            "source": "fallback", "license": "bundled", "title": fn})
    return out[:limit] if limit else out


def crawl(query="person full body standing", limit=40, out_dir=RAW_DIR):
    ensure_dir(out_dir)
    candidates = []
    for fetcher in (from_wikimedia, from_duckduckgo):
        if len([c for c in candidates]) >= limit * 2:
            break
        got = fetcher(query, limit)
        log.info("%s -> %d candidates", fetcher.__name__, len(got))
        candidates += got

    manifest, saved = [], 0
    for cand in candidates:
        if saved >= limit:
            break
        try:
            if cand["url"].startswith("file://"):
                content = open(cand["url"][7:], "rb").read()
            else:
                content = requests.get(cand["url"], headers=HEADERS, timeout=20).content
        except Exception:
            continue
        dst = osp.join(out_dir, f"img_{saved:04d}.jpg")
        if _save_image(content, dst):
            cand["path"] = dst
            manifest.append(cand)
            saved += 1
        time.sleep(0.05)

    # Guarantee a non-empty dataset.
    if saved == 0:
        log.info("No network images; copying bundled fallback assets.")
        for cand in use_fallback(limit):
            dst = osp.join(out_dir, f"img_{saved:04d}.jpg")
            try:
                Image.open(cand["url"][7:]).convert("RGB").save(dst, "JPEG")
                cand["path"] = dst
                manifest.append(cand)
                saved += 1
            except Exception:
                pass

    save_json({"query": query, "count": saved, "images": manifest},
              osp.join(out_dir, "manifest.json"))
    log.info("Saved %d images to %s", saved, out_dir)
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Crawl person images.")
    ap.add_argument("--query", default="person full body standing")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    crawl(args.query, args.limit)
