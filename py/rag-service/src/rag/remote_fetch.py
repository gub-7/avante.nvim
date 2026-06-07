"""Remote resource fetching utilities."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

import httpx
from libs.logger import logger

http_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
}


def is_remote_resource_exists(url: str) -> bool:
    """Check if a URL exists."""
    try:
        response = httpx.head(url, headers=http_headers)
        return response.status_code in {
            httpx.codes.OK,
            httpx.codes.MOVED_PERMANENTLY,
            httpx.codes.FOUND,
        }
    except (OSError, ValueError, RuntimeError) as e:
        logger.error("Error checking if URL exists %s: %s", url, e)
        return False


def fetch_markdown(url: str) -> str:
    """Fetch markdown content from a URL."""
    try:
        from markdownify import markdownify as md

        logger.info("Fetching markdown content from %s", url)
        response = httpx.get(url, headers=http_headers)
        if response.status_code == httpx.codes.OK:
            return md(response.text)
        return ""
    except (OSError, ValueError, RuntimeError) as e:
        logger.error("Error fetching markdown content %s: %s", url, e)
        return ""


def markdown_to_links(base_url: str, markdown: str) -> list[str]:
    """Extract links from markdown content."""
    links = []
    seek = {base_url}
    parsed_url = urlparse(base_url)
    domain = parsed_url.netloc
    scheme = parsed_url.scheme
    for match in re.finditer(r"\[(.*?)\]\((.*?)\)", markdown):
        url = match.group(2)
        if not url.startswith(scheme):
            url = urljoin(base_url, url)
        if urlparse(url).netloc != domain:
            continue
        if url in seek:
            continue
        seek.add(url)
        links.append(url)
    return links

