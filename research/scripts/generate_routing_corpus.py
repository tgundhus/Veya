"""Generate the reviewed cross-SDK routing contract fixtures (no models or network)."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from veya.lang import ROUTING_CONTRACT_VERSION, analyse, latin_profile, state_text  # noqa: E402
from veya._routing_unicode import TABLE_SHA256, VERSION  # noqa: E402
from veya.router import Router  # noqa: E402


NOTE = "Please check the attached invoice and let me know what happened. "
PROSE = {
    "en": "Please review this request and tell me what the customer needs today",
    "fr": "Bonjour je veux savoir pourquoi nous avons deux factures pour cette commande",
    "de": "Bitte prüfen Sie meine Rechnung weil ich heute eine Erstattung brauche",
    "pt": "Quero o meu dinheiro de volta agora mesmo porque já esperei demais",
    "es": "Quiero cancelar mi pedido porque todavía no tengo ninguna respuesta",
    "it": "Vorrei annullare il mio ordine per favore adesso grazie mille",
    "sv": "Jag behöver hjälp med min faktura och vill ha en återbetalning",
    "ro": "Vreau să verific această factură pentru că nu este corectă",
    "bn": "ami ekta ticket khulsi kalke theke payment hocche na ekhon ki korte parbo",
    "az": "ödənişim iki dəfə tutulub amma heç bir təsdiq almadım nə etməliyəm",
    "ru": "пожалуйста проверьте мою оплату и верните деньги за заказ",
    "zh": "请检查我的付款并退还重复收取的费用",
    "ja": "請求書を確認して重複した支払いを返金してください",
    "ar": "يرجى التحقق من فاتورتي وإعادة المبلغ المدفوع مرتين",
    "hi": "कृपया मेरा भुगतान जांचें और दो बार लिया गया शुल्क वापस करें",
    "ta": "தயவுசெய்து எனது கட்டணத்தை சரிபார்த்து பணத்தை திருப்பித் தரவும்",
}


def expand(value):
    """Fixture-only shorthand keeps long boundary examples small and reviewable."""
    if isinstance(value, dict):
        if set(value) == {"$repeat"}:
            text, count = value["$repeat"]
            return text * count
        if set(value) == {"$concat"}:
            return "".join(expand(item) for item in value["$concat"])
        if set(value) == {"$bytes"}:
            return bytes(value["$bytes"])
        if set(value) == {"$pairs"}:
            return {key: expand(item) for key, item in value["$pairs"]}
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    return value


def normalize(value):
    aliases = {"script_profile": "scriptProfile", "is_english": "isEnglish",
               "language_undecided": "languageUndecided", "diacritic_rate": "diacriticRate",
               "non_latin_fraction": "nonLatinFraction", "mixed_segment": "mixedSegment",
               "leaf_index": "leafIndex", "english_hits": "englishHits", "looks_non_english": "looksNonEnglish"}
    if isinstance(value, dict):
        return {aliases.get(key, key): normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def cases():
    rows = []

    def add(name, state, **kwargs):
        rows.append({"id": name, "state": state, **kwargs})

    for name, text in PROSE.items():
        add("prose-" + name, text)
        add("buried-" + name, {"note": {"$repeat": [NOTE, 100]}, "customer": text})
    for name, state in [
        ("no-letters", "1234 !!! 😀"), ("empty", ""), ("null", None),
        ("primitives", [True, False, 49, 0.51, None]),
        ("keys-ignored", {PROSE["fr"]: "English ticket and the customer will call"}),
        ("numeric-object-order", {"$pairs": [["10", {"$repeat": [PROSE["fr"], 80]}],
                                                ["2", {"$repeat": [PROSE["de"], 80]}]]}),
        ("array-index-boundaries", {"$pairs": [["4294967295", "last"], ["01", "leading"],
                                                  ["4294967294", "index"], ["0", "zero"], ["-1", "minus"]]}),
        ("empty-leaves", ["", "", "😀", "", "é", "X"]),
        ("nested", {"request": [{"body": PROSE["pt"]}], "amount": 51}),
        ("prototype-key", {"$pairs": [["__proto__", PROSE["pt"]], ["constructor", "ignored key"]]}),
        ("deep-visible", [[[[[[PROSE["pt"]]]]]]]),
        ("deep-hidden", [[[[[[[PROSE["pt"]]]]]]]]),
        ("emoji-budget", {"$concat": [{"$repeat": ["😀", 3500]}, "\n", PROSE["pt"]]}),
        ("per-line-budget", {"note": {"$repeat": [NOTE, 100]},
                             "customer": {"$concat": [{"$repeat": ["😀", 4001]}, "\n", PROSE["pt"]]}}),
        ("leaf-name-only", {"note": {"$repeat": [NOTE, 100]}, "name": "José"}),
        ("unicode-code", {"note": {"$repeat": [NOTE, 100]}, "customer": "α( je nous pour merci"}),
        ("unicode-compound", {"note": {"$repeat": [NOTE, 100]}, "customer": "α/com je nous"}),
        ("uppercase-modifier", {"note": {"$repeat": [NOTE, 100]}, "customer": "ᴬ JE NOUS POUR MERCI"}),
        ("lowercase-mark", {"note": {"$repeat": [NOTE, 100]}, "customer": "JE NOUS POUR MERCI\u0345"}),
        ("combining-zero", NOTE + "я\u034f" * 10), ("enclosing-zero", NOTE + "я\u20dd" * 10),
        ("numeric-letter", "Ⅳ JE NOUS POUR MERCI"), ("titlecase", "ǅ JE NOUS POUR MERCI"),
        ("new-unicode-letter", "\u1c89" * 12), ("new-case-mapping", "\ua7cb" * 4 + " lösenord tack"),
        ("turkish-dotted-i", "İ je nous para"), ("stopword-drift", "where je nous para"),
        ("shared-only", "la un e que"), ("collision", "Do more, do less"),
        ("acronyms", "MON DES EST LA"), ("shouted-prose", PROSE["pt"].upper()),
        ("whitespace-controls", NOTE + "\x1c" + PROSE["pt"]),
        ("bom-preserved", "\ufeff" + PROSE["pt"]), ("nul", "the\x00and\x00please\x00review"),
        ("lone-surrogate", "the \ud800 and \udfff please review"),
        ("no-normalization", "é e\u0301 😀"), ("dotted-identifiers", "com.example.net user@os.com the invoice"),
        ("unknown-vietnamese", "Cảm ơn bạn nhiều nhẻ"),
        ("path-too-long", {"x" * 129: {"note": {"$repeat": [NOTE, 100]}, "customer": PROSE["pt"]}}),
        ("bytes-bom", {"$bytes": [239, 187, 191, 65, 66, 67]}),
        ("bytes-invalid", {"$bytes": [255, 254]}),
        ("bytes-unicode", {"$bytes": list(PROSE["pt"].encode())}),
    ]:
        add(name, state)
    for code in ("en-US", "fr", "pt_BR.UTF-8", "C.UTF-8", "POSIX", "und", "  ", "de\x00", "x'\"", "\u1c89",
                 "\ufeffen", "en\ufeff", "\x1cen\x1c", "\x85en\x85", "\U0001e030", "\U0002ebf0"):
        add("explicit-lang-" + repr(code), PROSE["en"], options={"lang": code})
    for override in ({"model": " ML "}, {"task": "typed_decisions"}, {"lang_guess": "ro"},
                     {"lang_guess": "\ufeffen"}, {"lang_guess": "\x1cen"}, {"model": "\x1cen\x1c"},
                     {"lang": "en", "lang_guess": "fr"}, {"model": "english", "lang": "fr"}):
        add("precedence-" + str(len(rows)), PROSE["pt"], options=override)
    add("configured-default", "123 !!!", router={"default": "multilingual"})
    add("configured-hint", PROSE["en"], router={"lang_guess": "de"})
    questions = {name: {"type": "noul", "instructions": "Is this true?"}
                 for name in ("action", "category", "churn_risk", "needs_human", "urgency")}
    add("workflow", PROSE["en"], questions=questions, router={"auto_task_detection": True})
    return rows


def expected(row):
    state = expand(row["state"])
    router = Router(**row.get("router", {}))
    options = row.get("options", {})
    return {
        "text": {str(budget): state_text(state, budget) for budget in (-1, 0, 1, 2, 7, 31, 4000)},
        "analysis": normalize(analyse(state)),
        "explained": normalize(analyse(state, explain=True)),
        "profile": normalize(latin_profile(state_text(state))),
        "route": normalize(dict(router.route(state, row.get("questions", {}), explain=True, **options))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify existing expectations without rewriting")
    args = parser.parse_args()
    rows = cases()
    for row in rows:
        row["expected"] = expected(row)
    artifact = {"schema": "veya-routing-golden-v1", "contract": ROUTING_CONTRACT_VERSION,
                "unicode_version": VERSION, "unicode_table_sha256": TABLE_SHA256, "cases": rows}
    path = ROOT / "tests/fixtures/routing-parity-v1.json"
    data = json.dumps(artifact, ensure_ascii=True, indent=2) + "\n"
    if args.check:
        assert path.read_text(encoding="utf-8") == data, "routing golden contract differs; inspect before updating"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
    print("%d shared routing fixtures %s" % (len(rows), "verified" if args.check else "written"))


if __name__ == "__main__":
    main()
