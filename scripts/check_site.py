"""Check the built documentation's public identity and crawlable entry points."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree


class Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.canonical = None
        self.description = None
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for name in ("href", "src"):
            value = attrs.get(name) or ""
            if value:
                self.links.append(value)
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical = attrs.get("href")
        if tag == "meta" and attrs.get("name") == "description":
            self.description = attrs.get("content")


def check_site(directory):
    origin = "https://tgundhus.github.io/Veya/"
    expected = ("", "integration/", "production/", "consistency/", "reports/production-review/")
    prefix = urlsplit(origin).path
    for page in directory.rglob("*.html"):
        metadata = Metadata()
        metadata.feed(page.read_text(encoding="utf-8"))
        page_url = origin + page.relative_to(directory).as_posix()
        for link in metadata.links:
            target = urlsplit(urljoin(page_url, link))
            if target.netloc != urlsplit(origin).netloc:
                continue
            assert target.path.startswith(prefix), (page, link, "link escapes site prefix")
            local_path = directory / unquote(target.path[len(prefix):])
            assert local_path.exists(), (page, link, "missing local link target")
    for route in expected:
        page = directory / route / "index.html"
        metadata = Metadata()
        metadata.feed(page.read_text(encoding="utf-8"))
        assert metadata.canonical == origin + route, (page, metadata.canonical)
        assert metadata.description, "missing search description: %s" % page
    sitemap = ElementTree.parse(directory / "sitemap.xml")
    urls = [node.text for node in sitemap.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    assert urls and all(url.startswith(origin) for url in urls), "incorrect sitemap origin"
    assert set(origin + route for route in expected).issubset(urls), "missing sitemap entry points"
    robots = (directory / "robots.txt").read_text(encoding="utf-8")
    assert "Sitemap: " + origin + "sitemap.xml" in robots, "incorrect robots sitemap"
    assert "Disallow: /" not in robots, "site blocks crawling"
    print("Documentation identity and crawl checks passed (%d sitemap URLs)." % len(urls))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("site"))
    check_site(parser.parse_args().site)
