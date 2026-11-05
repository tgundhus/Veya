"""Freeze the Unicode properties used by Python and TypeScript language routing.

Run with Python 3.12 (Unicode 15.0.0):
    python research/scripts/generate_routing_unicode.py

No network or external Unicode package is required. Python's Unicode implementation
is the oracle, including the contextual Final_Sigma rule used by str.lower().
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
VERSION = "15.0.0"
PROPERTIES = ("ALPHA", "WORD", "ALNUM", "UPPER", "LOWER", "TITLE", "COMBINING", "SPACE", "CASED", "IGNORABLE", "PRINTABLE")


def collect():
    if unicodedata.unidata_version != VERSION:
        raise RuntimeError("generate routing tables with Unicode %s, found %s" %
                           (VERSION, unicodedata.unidata_version))
    properties = {name: [] for name in PROPERTIES}
    lower = {}
    for cp in range(0x110000):
        ch = chr(cp)
        category = unicodedata.category(ch)
        upper, small = ch.isupper(), ch.islower()
        cased = upper or small or category == "Lt"
        # Final_Sigma skips Case_Ignorable characters. Probe opposite following
        # contexts for cased and uncased input so the oracle distinguishes them.
        ignorable = (("A\u03a3" + ch + " ").lower()[1] == "\u03c2" if cased else
                     ("A\u03a3" + ch + "A").lower()[1] == "\u03c3")
        values = (ch.isalpha(), ch.isalnum() and category != "Nd", ch.isalnum(),
                  upper, small, category == "Lt", bool(unicodedata.combining(ch)),
                  ch.isspace(), cased, ignorable, ch.isprintable())
        for name, present in zip(PROPERTIES, values):
            if present:
                properties[name].append(cp)
        lowered = ch.lower()
        if lowered != ch:
            lower[cp] = lowered
    return properties, lower


def ranges(points):
    result = []
    for cp in points:
        if result and cp == result[-1][1] + 1:
            result[-1][1] = cp
        else:
            result.append([cp, cp])
    return result


def source(points, typescript=False):
    def escape(cp):
        if typescript:
            return "\\u{%x}" % cp
        return ("\\u%04x" if cp <= 0xffff else "\\U%08x") % cp
    return "".join(escape(lo) if lo == hi else escape(lo) + "-" + escape(hi)
                   for lo, hi in ranges(points))


def mapping_lines(lower, typescript=False):
    entries = ["%s: %s" % (hex(cp), json.dumps(value, ensure_ascii=True) if typescript else ascii(value))
               for cp, value in lower.items()]
    return "\n".join("    " + ", ".join(entries[i:i + 8]) + ","
                     for i in range(0, len(entries), 8))


def verification_hashes(properties, lower):
    masks = bytearray(0x110000 * 2)
    for index, name in enumerate(PROPERTIES):
        offset, flag = (0, 1 << index) if index < 8 else (1, 1 << (index - 8))
        for cp in properties[name]:
            masks[cp * 2 + offset] |= flag
    lowered = hashlib.sha256()
    for cp in range(0x110000):
        lowered.update(lower.get(cp, chr(cp)).encode("utf-16-le", "surrogatepass"))
    return hashlib.sha256(masks).hexdigest(), lowered.hexdigest()


def python_module(properties, lower, identity, hashes):
    lines = [
        '"""Generated Unicode 15.0.0 routing semantics; regenerate with research/scripts/generate_routing_unicode.py."""',
        "import re", "import unicodedata", "",
        'VERSION = "%s"' % VERSION,
        'TABLE_SHA256 = "%s"' % identity,
        'PROPERTY_MASK_SHA256 = "%s"' % hashes[0],
        'LOWERCASE_SHA256 = "%s"' % hashes[1],
    ]
    for name in PROPERTIES:
        lines.append('%s_SOURCE = r"%s"' % (name, source(properties[name])))
    lines += ["", "ALPHA = re.compile(\"[\" + ALPHA_SOURCE + \"]\")",
              "WORD = re.compile(\"[\" + WORD_SOURCE + \"]+\")"]
    for name in ("UPPER", "LOWER", "TITLE", "COMBINING", "SPACE", "CASED", "IGNORABLE", "PRINTABLE"):
        lines.append('%s = re.compile("[" + %s_SOURCE + "]")' % (name, name))
    lines += ["", "_LOWER_MAP = {", mapping_lines(lower), "}", "",
              '_NATIVE_VERSION = unicodedata.unidata_version == VERSION', ""]
    lines.append('''def unicode_isupper(text: str) -> bool:
    """Unicode 15 string isupper: cased input, with no lower or titlecase letters."""
    if _NATIVE_VERSION or text.isascii():
        return text.isupper()
    return bool(UPPER.search(text)) and LOWER.search(text) is None and TITLE.search(text) is None


def unicode_lower(text: str) -> str:
    """Unicode 15 lowercase, including context-sensitive Greek Final_Sigma."""
    if _NATIVE_VERSION or text.isascii():
        return text.lower()
    if "\\u03a3" not in text:
        return text.translate(_LOWER_MAP)
    following = False
    following_cased = [False] * len(text)
    for index in range(len(text) - 1, -1, -1):
        ch = text[index]
        following_cased[index] = following
        if IGNORABLE.fullmatch(ch) is None:
            following = CASED.fullmatch(ch) is not None
    before = False
    result = []
    for index, ch in enumerate(text):
        if ch == "\\u03a3":
            result.append("\\u03c2" if before and not following_cased[index] else "\\u03c3")
        else:
            result.append(_LOWER_MAP.get(ord(ch), ch))
        if IGNORABLE.fullmatch(ch) is None:
            before = CASED.fullmatch(ch) is not None
    return "".join(result)
''')
    lines.append(r'''def unicode_repr(value) -> str:
    """Python string repr with Unicode 15 printability, independent of the host release."""
    if not isinstance(value, str) or _NATIVE_VERSION:
        return repr(value)
    quote = '"' if "'" in value and '"' not in value else "'"
    escaped = []
    for ch in value:
        if ch == "\\" or ch == quote:
            escaped.append("\\" + ch)
        elif ch == "\t":
            escaped.append("\\t")
        elif ch == "\n":
            escaped.append("\\n")
        elif ch == "\r":
            escaped.append("\\r")
        elif PRINTABLE.fullmatch(ch) is not None:
            escaped.append(ch)
        else:
            cp = ord(ch)
            escaped.append(("\\x%02x" if cp <= 255 else "\\u%04x" if cp <= 65535 else "\\U%08x") % cp)
    return quote + "".join(escaped) + quote
''')
    return "\n".join(lines)


def typescript_module(properties, lower, identity, hashes):
    lines = [
        "// Generated Unicode 15.0.0 routing semantics; regenerate with research/scripts/generate_routing_unicode.py.",
        'export const VERSION = "%s";' % VERSION,
        'export const TABLE_SHA256 = "%s";' % identity,
        'export const PROPERTY_MASK_SHA256 = "%s";' % hashes[0],
        'export const LOWERCASE_SHA256 = "%s";' % hashes[1],
    ]
    for name in PROPERTIES:
        lines.append("export const %s_SOURCE = %s;" % (name, json.dumps(source(properties[name], True))))
    lines.append("")
    for name in ("ALPHA", "UPPER", "LOWER", "TITLE", "COMBINING", "SPACE", "CASED", "IGNORABLE", "PRINTABLE"):
        lines.append('export const %s_RE = new RegExp("[" + %s_SOURCE + "]", "u");' % (name, name))
    lines += ["", "const LOWER_MAP: Record<number, string> = {", mapping_lines(lower, True), "};", ""]
    lines.append('const LOWER_CHANGE_RE = new RegExp("[" + %s + "]", "gu");' % json.dumps(source(lower, True)))
    lines.append('''export function unicodeIsUpper(text: string): boolean {
  return UPPER_RE.test(text) && !LOWER_RE.test(text) && !TITLE_RE.test(text);
}

/** Unicode 15 lowercase, independent of the host engine's Unicode release. */
export function unicodeLower(text: string): string {
  if (!/[^\\x00-\\x7f]/.test(text)) return text.toLowerCase();
  if (!text.includes("\\u03a3")) {
    return text.replace(LOWER_CHANGE_RE, (ch) => LOWER_MAP[ch.codePointAt(0)!]);
  }
  const chars = Array.from(text);
  const followingCased: boolean[] = new Array(chars.length);
  let following = false;
  for (let index = chars.length - 1; index >= 0; index--) {
    const ch = chars[index];
    followingCased[index] = following;
    if (!IGNORABLE_RE.test(ch)) following = CASED_RE.test(ch);
  }
  let before = false;
  let lowered = "";
  for (let index = 0; index < chars.length; index++) {
    const ch = chars[index];
    lowered += ch === "\\u03a3"
      ? before && !followingCased[index] ? "\\u03c2" : "\\u03c3"
      : LOWER_MAP[ch.codePointAt(0)!] ?? ch;
    if (!IGNORABLE_RE.test(ch)) before = CASED_RE.test(ch);
  }
  return lowered;
}
''')
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify checked-in tables and exhaustively test the frozen Python fallback")
    parser.add_argument("--check-node", action="store_true",
                        help="also verify every generated TypeScript property/lowercase with Node 22.18+")
    args = parser.parse_args()
    args.check = args.check or args.check_node
    properties, lower = collect()
    identity = hashlib.sha256(json.dumps({"version": VERSION, "properties": properties, "lower": lower},
                                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    hashes = verification_hashes(properties, lower)
    outputs = {
        ROOT / "laya/_routing_unicode.py": python_module(properties, lower, identity, hashes),
        ROOT / "laya-ts/src/routing-unicode.ts": typescript_module(properties, lower, identity, hashes),
    }
    for path, content in outputs.items():
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise AssertionError("stale routing Unicode output: %s" % path.relative_to(ROOT))
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
            print(path.relative_to(ROOT), len(content), "bytes")
    if args.check:
        verify_python()
    if args.check_node:
        verify_node(hashes)
    print("Unicode", VERSION, "table SHA256", identity)


def verify_python():
    spec = importlib.util.spec_from_file_location("routing_unicode", ROOT / "laya/_routing_unicode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._NATIVE_VERSION = False           # exercise compatibility with other Python Unicode releases
    for cp in range(0x110000):
        ch = chr(cp)
        category = unicodedata.category(ch)
        expected = (ch.isalpha(), ch.isalnum() and category != "Nd", ch.isupper(), ch.islower(),
                    category == "Lt", bool(unicodedata.combining(ch)), ch.isspace(), ch.isprintable())
        actual = tuple(pattern.fullmatch(ch) is not None for pattern in
                       (module.ALPHA, module.WORD, module.UPPER, module.LOWER,
                        module.TITLE, module.COMBINING, module.SPACE, module.PRINTABLE))
        if expected != actual or module.unicode_lower(ch) != ch.lower() or module.unicode_repr(ch) != repr(ch):
            raise AssertionError("frozen routing property/lowercase mismatch: U+%04X" % cp)
    rng = random.Random(1500)
    pool = ("A", "a", "\u03a3", "\u03b1", "\u03c3", "\u03c2", "'", "\u034f", "\u0345",
            "\u1d2c", "\u2160", "\u01c5", "\u0130", " ", ".", "1", "_", "\u0301")
    for _ in range(10000):
        text = "".join(rng.choices(pool, k=rng.randrange(40)))
        if module.unicode_lower(text) != text.lower() or module.unicode_isupper(text) != text.isupper():
            raise AssertionError("frozen routing contextual casing mismatch: %r" % text)
    print("Verified 1,114,112 code points and 10,000 contextual casing fixtures")


def verify_node(hashes):
    rng = random.Random(1500)
    pool = ("A", "a", "\u03a3", "\u03b1", "\u03c3", "\u03c2", "'", "\u034f", "\u0345",
            "\u1d2c", "\u2160", "\u01c5", "\u0130", " ", ".", "1", "_", "\u0301")
    texts = ["".join(rng.choices(pool, k=rng.randrange(40))) for _ in range(10000)]
    code = r'''
import fs from "node:fs";
import crypto from "node:crypto";
import { pathToFileURL } from "node:url";
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const table = await import(pathToFileURL(input.module).href);
const patterns = input.properties.map(name => new RegExp("[" + table[name + "_SOURCE"] + "]", "u"));
const masks = Buffer.alloc(0x110000 * 2);
const lower = crypto.createHash("sha256");
for (let cp = 0; cp < 0x110000; cp++) {
  const ch = String.fromCodePoint(cp);
  let mask = 0;
  patterns.forEach((pattern, index) => { if (pattern.test(ch)) mask |= 1 << index; });
  masks.writeUInt16LE(mask, cp * 2);
  lower.update(Buffer.from(table.unicodeLower(ch), "utf16le"));
}
console.log(JSON.stringify({
  properties: crypto.createHash("sha256").update(masks).digest("hex"),
  lower: lower.digest("hex"),
  contexts: input.texts.map(text => [table.unicodeLower(text), table.unicodeIsUpper(text)]),
}));
'''
    payload = {"module": str(ROOT / "laya-ts/src/routing-unicode.ts"),
               "properties": PROPERTIES, "texts": texts}
    result = subprocess.run(["node", "--input-type=module", "-e", code], cwd=ROOT,
                            input=json.dumps(payload), encoding="utf-8", capture_output=True, check=True)
    actual = json.loads(result.stdout)
    if (actual["properties"], actual["lower"]) != hashes:
        raise AssertionError("generated TypeScript routing properties/lowercase differ from Unicode 15")
    if actual["contexts"] != [[text.lower(), text.isupper()] for text in texts]:
        raise AssertionError("generated TypeScript contextual casing differs from Unicode 15")
    print("Verified all generated TypeScript properties/lowercases and 10,000 contextual casing fixtures")


if __name__ == "__main__":
    main()
