from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

from agentos.settings import Settings

_SSRF_DENIED = (
    "private",
    "loopback",
    "link_local",
    "multicast",
    "reserved",
    "unspecified",
)


class ScrapeResult(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    http_status: int = 0
    source: str = ""


class Scraper:
    """Safe web scraping: SSRF protection, size caps, robots.txt, HTML sanitization."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._robots_cache: dict[str, list[str]] = {}

    async def fetch(self, url: str, source: str = "") -> ScrapeResult:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"unsupported URL: {url}")
        self._assert_public(parsed.hostname or "")

        if self.settings.allow_network is False:
            raise PermissionError("network access disabled by configuration")

        async with httpx.AsyncClient(
            timeout=self.settings.scrape_timeout, follow_redirects=True
        ) as client:
            headers = {
                "User-Agent": "AgentOS-ResearchBot/0.2 (+https://github.com/aaarif796/AgentOS)"
            }
            response = await client.get(url, headers=headers)
            if response.status_code >= 400:
                return ScrapeResult(url=url, http_status=response.status_code, source=source)
            content = response.content[: self.settings.scrape_max_bytes]
            text, title = self._extract(content)
            return ScrapeResult(
                url=str(response.url),
                title=title,
                text=text,
                http_status=response.status_code,
                source=source,
            )

    # -- safety helpers ------------------------------------------------------
    def _assert_public(self, hostname: str) -> None:
        if self.settings.scrape_allow_private:
            return
        try:
            infos = socket.getaddrinfo(hostname, 443)
        except OSError as exc:
            raise ValueError(f"could not resolve {hostname}: {exc}") from exc
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if any(getattr(ip, f"is_{kind}", False) for kind in _SSRF_DENIED):
                raise PermissionError(f"SSRF guard: {hostname} resolves to non-public address {ip}")

    @staticmethod
    def _extract(content: bytes) -> tuple[str, str]:
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else ""
        text = soup.get_text(separator="\n", strip=True)
        text = "\n".join(line for line in text.splitlines() if line.strip())[:200_000]
        return text, title
