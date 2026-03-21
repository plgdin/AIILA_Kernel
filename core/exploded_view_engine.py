from __future__ import annotations

import json
import os
import re
import shutil
import threading
import hashlib
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import quote_plus, urljoin, urlparse

import cv2
import numpy as np
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "unknown_device"


def _cache_slug(model_name: str) -> str:
    normalized = (model_name or "").strip().lower()
    if not normalized:
        return "unknown_device"

    normalized = normalized.replace("+", " plus ")
    normalized = re.sub(r"\bapple\b", " ", normalized)
    if _looks_like_samsung_model(normalized):
        normalized = re.sub(r"\bsamsung\b", " ", normalized)

    tokens = re.findall(r"[a-z0-9]+", normalized)
    ignored_tokens = {
        "5g",
        "repair",
        "manual",
        "teardown",
        "exploded",
        "view",
        "internal",
    }
    tokens = [token for token in tokens if token not in ignored_tokens]
    if not tokens:
        return "unknown_device"
    return "_".join(tokens)


def _clean_response_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _pick_src_from_srcset(srcset: str) -> str:
    parts = [chunk.strip() for chunk in srcset.split(",") if chunk.strip()]
    if not parts:
        return ""
    return parts[-1].split()[0]


def _looks_like_samsung_model(model_name: str) -> bool:
    model_name = (model_name or "").lower()
    samsung_tokens = (
        "samsung",
        "galaxy",
        "z flip",
        "z fold",
        "note",
    )
    return any(token in model_name for token in samsung_tokens)


def _expand_search_terms(model_name: str) -> list[str]:
    cleaned = (model_name or "").strip()
    if not cleaned:
        return []

    terms = [cleaned]
    lowered = cleaned.lower()
    if _looks_like_samsung_model(cleaned) and "samsung" not in lowered:
        terms.append(f"Samsung {cleaned}")
    if "iphone" in lowered and "apple" not in lowered:
        terms.append(f"Apple {cleaned}")

    deduped = []
    seen = set()
    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(term)
    return deduped


def _source_hint_for_model(model_name: str) -> str:
    lowered = (model_name or "").lower()
    if _looks_like_samsung_model(model_name):
        return (
            "This is a Samsung Galaxy phone. Prioritize exact-model internal photos, "
            "self-repair pages, service-manual imagery, parts breakdowns, and teardown "
            "guides from Samsung, Samsung-authorized parts sources, or reputable repair sites."
        )
    if "iphone" in lowered:
        return (
            "This is an Apple iPhone. Prioritize Apple's repair manuals, official internal "
            "views, and reputable teardown pages for the exact iPhone model."
        )
    return (
        "Prioritize official repair manuals, vendor repair portals, and reputable teardown "
        "or repair guides for the exact device."
    )


@dataclass
class ImageRecord:
    path: str
    caption: str
    source_url: str


class _ImageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.page_title = ""
        self._in_title = False
        self.image_candidates: list[dict] = []

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if tag == "title":
            self._in_title = True
            return

        if tag == "meta":
            prop = (attr.get("property") or attr.get("name") or "").lower()
            if prop in {"og:image", "twitter:image", "twitter:image:src"}:
                src = attr.get("content", "").strip()
                if src:
                    self.image_candidates.append({
                        "src": src,
                        "alt": "",
                        "title": prop,
                    })
            return

        if tag != "img":
            return

        src = (
            attr.get("src")
            or attr.get("data-src")
            or attr.get("data-lazy-src")
            or attr.get("data-original")
            or attr.get("data-image")
            or ""
        ).strip()
        if not src and attr.get("srcset"):
            src = _pick_src_from_srcset(attr["srcset"])

        if src:
            self.image_candidates.append({
                "src": src,
                "alt": attr.get("alt", "").strip(),
                "title": attr.get("title", "").strip(),
            })

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.page_title += data.strip()


class ExplodedViewEngine:
    MAX_IMAGES = 6

    def __init__(self, cache_dir: str = "output/exploded_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.client = genai.Client(api_key=API_KEY) if API_KEY else None
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/132.0.0.0 Safari/537.36"
            )
        })

        self._lock = threading.RLock()
        self._items: list[ImageRecord] = []
        self._index = 0
        self._model_name = ""
        self._category = ""
        self._current_image: Optional[np.ndarray] = None
        self._current_path = ""

    def clear(self):
        with self._lock:
            self._items.clear()
            self._index = 0
            self._model_name = ""
            self._category = ""
            self._current_image = None
            self._current_path = ""

    def cache_root_for_model(self, model_name: str) -> str:
        return os.path.join(self.cache_dir, _cache_slug(model_name))

    def load_for_model(
        self,
        model_name: str,
        category: str = "",
        refresh: bool = False,
    ) -> tuple[bool, str]:
        model_name = (model_name or "").strip()
        if not model_name:
            return False, "No scanned model available."

        with self._lock:
            self._items.clear()
            self._index = 0
            self._model_name = model_name
            self._category = category or ""
            self._current_image = None
            self._current_path = ""

        cache_root = self.cache_root_for_model(model_name)
        if refresh and os.path.isdir(cache_root):
            try:
                shutil.rmtree(cache_root)
            except Exception:
                pass
        os.makedirs(cache_root, exist_ok=True)
        manifest_path = os.path.join(cache_root, "manifest.json")

        items = self._load_manifest(manifest_path)
        if not items:
            items = self._discover_and_cache(model_name, category, cache_root, manifest_path)

        if not items:
            return False, f"No exploded-view images found for {model_name}."

        with self._lock:
            self._items = items
            self._index = 0
            self._load_current_image_locked()

        return True, f"Exploded view ready for {model_name}."

    def has_images(self) -> bool:
        with self._lock:
            return bool(self._items)

    def next_image(self) -> bool:
        with self._lock:
            if len(self._items) <= 1:
                return False
            self._index = (self._index + 1) % len(self._items)
            self._load_current_image_locked()
            return True

    def previous_image(self) -> bool:
        with self._lock:
            if len(self._items) <= 1:
                return False
            self._index = (self._index - 1) % len(self._items)
            self._load_current_image_locked()
            return True

    def get_view_state(self) -> dict:
        with self._lock:
            if not self._items or self._current_image is None:
                return {
                    "visible": False,
                    "image": None,
                    "caption": "",
                    "index": 0,
                    "total": 0,
                    "model_name": self._model_name,
                    "source_url": "",
                }

            item = self._items[self._index]
            return {
                "visible": True,
                "image": self._current_image.copy(),
                "caption": item.caption,
                "index": self._index + 1,
                "total": len(self._items),
                "model_name": self._model_name,
                "source_url": item.source_url,
            }

    def _load_manifest(self, manifest_path: str) -> list[ImageRecord]:
        if not os.path.exists(manifest_path):
            return []

        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return []

        items = []
        for row in payload.get("images", []):
            path = row.get("path", "")
            if path and os.path.exists(path):
                items.append(ImageRecord(
                    path=path,
                    caption=row.get("caption", "Internal view"),
                    source_url=row.get("source_url", ""),
                ))
        return items[:self.MAX_IMAGES]

    def _discover_and_cache(
        self,
        model_name: str,
        category: str,
        cache_root: str,
        manifest_path: str,
    ) -> list[ImageRecord]:
        pages = self._discover_source_pages(model_name, category)
        if not pages:
            pages = self._fallback_search_pages(model_name)

        seen_urls = set()
        items: list[ImageRecord] = []
        for page in pages:
            page_url = page.get("url", "").strip()
            if not page_url or page_url in seen_urls:
                continue
            seen_urls.add(page_url)
            title = page.get("title", "").strip()
            found = self._extract_page_images(page_url, title, model_name, cache_root)
            for item in found:
                items.append(item)
                if len(items) >= self.MAX_IMAGES:
                    break
            if len(items) >= self.MAX_IMAGES:
                break

        if not items:
            return []

        payload = {
            "model_name": model_name,
            "images": [
                {"path": item.path, "caption": item.caption, "source_url": item.source_url}
                for item in items
            ],
        }
        try:
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception:
            pass

        return items

    def _discover_source_pages(self, model_name: str, category: str) -> list[dict]:
        if self.client is None:
            return []

        response = None
        search_terms = _expand_search_terms(model_name) or [model_name]
        search_list = ", ".join(f'"{term}"' for term in search_terms)
        prompt = f"""
Find web pages with exploded view, teardown, internal component photos, or repair manual imagery
for the device "{model_name}".
Category: "{category or 'unknown'}".
Search aliases: {search_list}.
{_source_hint_for_model(model_name)}
Prioritize official manuals and reputable teardown/repair sites.
Return strict JSON only in this format:
[
  {{
    "url": "https://...",
    "title": "Page title",
    "reason": "why this page matters"
  }}
]
Limit to 5 pages.
"""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            text = _clean_response_text(getattr(response, "text", "") or "")
            pages = json.loads(text) if text else []
            if isinstance(pages, list):
                cleaned = []
                for row in pages:
                    if isinstance(row, dict) and row.get("url"):
                        cleaned.append({
                            "url": row.get("url", ""),
                            "title": row.get("title", ""),
                            "reason": row.get("reason", ""),
                        })
                if cleaned:
                    return cleaned[:5]
        except Exception:
            pass

        chunks = []
        try:
            candidates = getattr(response, "candidates", [])  # type: ignore[name-defined]
            if candidates:
                meta = getattr(candidates[0], "grounding_metadata", None)
                chunks = getattr(meta, "grounding_chunks", []) or []
        except Exception:
            chunks = []

        cleaned = []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            url = getattr(web, "uri", "") if web else ""
            title = getattr(web, "title", "") if web else ""
            if url:
                cleaned.append({"url": url, "title": title, "reason": "grounded search"})
        return cleaned[:5]

    def _fallback_search_pages(self, model_name: str) -> list[dict]:
        queries = _expand_search_terms(model_name)
        primary_query = queries[0] if queries else model_name
        encoded_query = quote_plus(primary_query)

        if _looks_like_samsung_model(model_name):
            return [
                {
                    "url": f"https://www.samsung.com/us/search/searchMain/?listType=g&searchTerm={quote_plus(primary_query + ' self repair')}",
                    "title": "Samsung Search Results",
                    "reason": "official Samsung search for self-repair material",
                },
                {
                    "url": f"https://samsungparts.com/search?q={encoded_query}",
                    "title": "Samsung Parts Search",
                    "reason": "Samsung parts catalog and assembly imagery",
                },
                {
                    "url": f"https://www.ifixit.com/Search?doctype=guide&query={encoded_query}",
                    "title": "iFixit Device Search",
                    "reason": "reputable teardown and repair guide search",
                },
            ]

        if "iphone" in model_name.lower():
            return [
                {
                    "url": "https://support.apple.com/repair/manuals",
                    "title": "Apple Repair Manuals",
                    "reason": "official Apple repair manual portal",
                }
            ]
        return []

    def _extract_page_images(
        self,
        page_url: str,
        page_title: str,
        model_name: str,
        cache_root: str,
    ) -> list[ImageRecord]:
        try:
            resp = self.session.get(page_url, timeout=20)
            resp.raise_for_status()
        except Exception:
            return []

        content_type = (resp.headers.get("content-type") or "").lower()
        if "image/" in content_type:
            item = self._download_candidate_image(
                page_url,
                page_title or model_name,
                page_url,
                cache_root,
                0,
            )
            return [item] if item else []

        if "text/html" not in content_type:
            return []

        parser = _ImageParser()
        try:
            parser.feed(resp.text)
        except Exception:
            return []

        page_title = page_title or parser.page_title or model_name
        candidates = []
        for idx, candidate in enumerate(parser.image_candidates):
            src = urljoin(page_url, candidate.get("src", ""))
            if not src.startswith("http"):
                continue
            score = self._score_candidate(src, candidate, page_url, page_title, model_name)
            if score <= 0:
                continue
            candidates.append((score, idx, src, candidate))

        candidates.sort(key=lambda row: row[0], reverse=True)

        items = []
        seen = set()
        for score, idx, src, candidate in candidates[:20]:
            if src in seen:
                continue
            seen.add(src)
            caption_bits = [
                candidate.get("alt", "").strip(),
                candidate.get("title", "").strip(),
                page_title.strip(),
            ]
            caption = " | ".join(bit for bit in caption_bits if bit) or page_title
            item = self._download_candidate_image(src, caption, page_url, cache_root, idx)
            if item:
                items.append(item)
            if len(items) >= 3:
                break

        return items

    def _score_candidate(
        self,
        src: str,
        candidate: dict,
        page_url: str,
        page_title: str,
        model_name: str,
    ) -> int:
        haystack = " ".join([
            src,
            candidate.get("alt", ""),
            candidate.get("title", ""),
            page_url,
            page_title,
        ]).lower()

        score = 0
        keywords = {
            "exploded": 7,
            "internal": 6,
            "inside": 5,
            "teardown": 7,
            "disassembly": 6,
            "repair": 4,
            "self-repair": 5,
            "self repair": 5,
            "service manual": 5,
            "battery": 3,
            "board": 3,
            "mainboard": 3,
            "motherboard": 3,
            "daughterboard": 3,
            "camera": 2,
            "parts": 3,
            "logic": 2,
            "iphone": 2,
            "samsung": 3,
            "galaxy": 3,
            "encompass": 3,
            "assembly": 2,
            "frame": 2,
            "flex": 2,
            "phone": 1,
        }
        for word, value in keywords.items():
            if word in haystack:
                score += value

        for token in re.findall(r"[a-z0-9]+", model_name.lower()):
            if len(token) > 1 and token in haystack:
                score += 2

        page_domain = urlparse(page_url).netloc.lower()
        image_domain = urlparse(src).netloc.lower()
        trusted_domains = {
            "support.apple.com": 6,
            "apple.com": 3,
            "samsung.com": 6,
            "samsungparts.com": 5,
            "encompass.com": 4,
            "ifixit.com": 3,
        }
        for domain, value in trusted_domains.items():
            if page_domain.endswith(domain) or image_domain.endswith(domain):
                score += value

        penalties = (
            "logo",
            "icon",
            "avatar",
            "sprite",
            "thumb",
            "banner",
            "ads",
            "lifestyle",
            "hero",
            "marketing",
        )
        if any(token in haystack for token in penalties):
            score -= 8

        if src.lower().endswith((".svg", ".gif")):
            score -= 6

        return score

    def _download_candidate_image(
        self,
        image_url: str,
        caption: str,
        source_url: str,
        cache_root: str,
        index: int,
    ) -> Optional[ImageRecord]:
        try:
            resp = self.session.get(image_url, timeout=20)
            resp.raise_for_status()
        except Exception:
            return None

        content_type = (resp.headers.get("content-type") or "").lower()
        if "image/" not in content_type and not image_url.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            return None

        arr = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        h, w = img.shape[:2]
        if min(h, w) < 220:
            return None

        if w > 1600:
            scale = 1600.0 / w
            img = cv2.resize(img, (1600, int(h * scale)), interpolation=cv2.INTER_AREA)

        digest = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:10]
        filename = os.path.join(cache_root, f"image_{index:02d}_{digest}.jpg")
        try:
            cv2.imwrite(filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        except Exception:
            return None

        return ImageRecord(path=filename, caption=caption[:180], source_url=source_url)

    def _load_current_image_locked(self):
        if not self._items:
            self._current_image = None
            self._current_path = ""
            return

        item = self._items[self._index]
        if self._current_path == item.path and self._current_image is not None:
            return

        img = cv2.imread(item.path)
        if img is None:
            self._current_image = None
            self._current_path = ""
            return

        self._current_image = img
        self._current_path = item.path
