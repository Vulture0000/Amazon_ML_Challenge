"""
erlib.py -- shared normalization / IO helpers for the Business Entity Resolution task.

Text canonicalization pipeline
------------------------------
1.  Unicode NFKD normalization ("INC" step) + combining-mark stripping, so that
    accented Latin text collapses to a single canonical form
    (  "Rte. de l'ete"  ->  "rte de l ete"  ).
2.  Script folding: any remaining non-Latin script (Devanagari, Kannada,
    Gujarati, Tamil, ...) is romanized so that the same business written in two
    scripts lands in a comparable space.
3.  Punctuation / symbol folding: "&" -> "and", "°"/"º" -> "no", dashes and
    dots -> space, everything non [a-z0-9] -> space.
4.  Domain-aware token canonicalization:
      * US address-type abbreviations  (st -> street, ave -> avenue, ...)
      * US / India state names        (connecticut -> ct, gujarat -> gj, ...)
      * French street types + regions (r -> rue, hauts de france -> ...)
      * ambiguous 2-letter tokens resolved by context
        ("10 Ct" -> court,  "Danbury, CT" -> connecticut)
5.  Two views of every field are produced:
      * ``*_norm`` : full canonical string  (used for char-level / fuzzy views)
      * ``*_core`` : legal-form / boilerplate tokens removed (used for word-level
                     TF-IDF, token overlap and blocking)
6.  Extra light-weight fields kept for the classifier: country code, numeric
    tokens, non-Latin-script flags and token counts.

Everything here is pure-Python and picklable so it can be shipped to worker
processes; there is no dependency on any external business database.
"""

import re
import unicodedata

from unidecode import unidecode

# --------------------------------------------------------------------------
# generic
# --------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_COMBINING = re.compile(r"[\u0300-\u036f]")
_CJK_LATIN_GREEK = re.compile(r"[\u0370-\u03ff\u0400-\u04ff]")

# unicode punctuation that carries meaning before folding
_DEGREE = re.compile(r"[\u00b0\u00ba\u00b0\u2070\u00b7]")
_DASH = re.compile(r"[\u2010-\u2015\u2212\u2013\u2014]")
_CURLY_APOS = re.compile(r"[\u2018\u2019\u02bc\u00b4\u0060]")


def has_non_latin(s: str) -> bool:
    """True when the string contains a character outside the Latin/ASCII range."""
    for ch in s:
        if ord(ch) > 0x24F:
            return True
    return False


def to_latin(s: str) -> str:
    """NFKD-normalize, drop combining marks, romanize leftover scripts."""
    if s.isascii():
        return s
    s = unicodedata.normalize("NFKD", s)
    s = _COMBINING.sub("", s)
    if s.isascii():
        return s
    s = unidecode(s)
    s = unicodedata.normalize("NFKD", s)
    return _COMBINING.sub("", s)


def fold_symbols(s: str) -> str:
    s = _DEGREE.sub(" no ", s)
    s = _DASH.sub(" ", s)
    s = _CURLY_APOS.sub("", s)
    s = s.replace("&", " and ").replace("+", " plus ")
    return s


# unidecode romanizes Devanagari/Kannada matras as doubled vowels
# ("praaivett", "maarketing"); collapsing them brings the romanization much
# closer to the Latin spelling of the same word ("private", "marketing").
_DOUBLE_VOWEL = re.compile(r"([aeiou])\1+")


def collapse_vowels(s: str) -> str:
    return _DOUBLE_VOWEL.sub(r"\1", s)


def squash(s: str) -> str:
    """lower-case and turn every non [a-z0-9] run into a single space."""
    return _NON_ALNUM.sub(" ", s.lower()).strip()


# --------------------------------------------------------------------------
# address canonicalization tables
# --------------------------------------------------------------------------

# street-type abbreviations -> canonical long form
STREET_ABBR = {
    "st": "street", "str": "street", "stret": "street",
    "rd": "road", "ave": "avenue", "av": "avenue", "aven": "avenue", "avenue": "avenue",
    "dr": "drive", "drv": "drive", "ln": "lane", "blvd": "boulevard", "blv": "boulevard",
    "hwy": "highway", "pkwy": "parkway", "pky": "parkway", "cir": "circle", "crcl": "circle",
    "pl": "place", "plz": "plaza", "sq": "square", "ter": "terrace", "terr": "terrace",
    "trl": "trail", "way": "way", "wy": "way", "hvn": "haven", "hts": "heights",
    "mnr": "manor", "mnt": "mount", "pt": "point", "rte": "route", "xing": "crossing",
    "aly": "alley", "alye": "alley", "bnd": "bend", "brg": "bridge", "clf": "cliff",
    "cres": "crescent", "est": "estate", "exp": "expressway", "ext": "extension",
    "fwy": "freeway", "gtwy": "gateway", "grn": "green", "hrbor": "harbor", "htl": "hotel",
    "in": "industrial", "jct": "junction", "lndng": "landing", "lf": "lookout",
    "lc": "lake", "mdw": "meadow", "nook": "nook", "orch": "orchard", "prt": "port",
    "rad": "radial", "res": "reservoir", "row": "row", "shoal": "shoal", "shr": "shore",
    "spur": "spur", "sta": "station", "strt": "street", "sum": "summit", "tpke": "turnpike",
    "vly": "valley", "vllg": "village", "vlg": "village", "walk": "walk",
    # french
    "r": "rue", "ch": "chemin", "che": "chemin", "imp": "impasse", "all": "allee",
    "bd": "boulevard", "avn": "avenue", "cours": "cours", "quai": "quai", "sent": "sentier",
    "rpt": "residence", "cite": "cite", "citee": "cite", "gal": "galerie", "plce": "place",
    # india
    "rdd": "road", "temple": "temple", "nagar": "nagar", "sec": "sector",
    "dist": "district", "div": "division",
}

# unit / secondary-address markers -> single canonical token
UNIT_ABBR = {
    "apt": "unit", "apartment": "unit", "ste": "unit", "suite": "unit",
    "fl": "unit", "flr": "unit", "floor": "unit", "rm": "unit", "room": "unit",
    "no": "unit", "num": "unit", "number": "unit", "hn": "unit", "hno": "unit",
    "kh": "unit", "khno": "unit", "shop": "unit", "shops": "unit", "office": "unit",
    "ofc": "unit", "bldg": "unit", "bldgno": "unit", "tower": "unit", "plot": "unit",
    "pvt": "unit", "part": "unit", "wing": "unit", "block": "unit", "blok": "unit",
    "k": "unit", "kb": "unit",
}

DIRECTIONS = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "northeast": "northeast", "northwest": "northwest",
    "southeast": "southeast", "southwest": "southwest",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky",
    "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy", "new york": "ny", "north carolina": "nc", "south carolina": "sc",
    "north dakota": "nd", "south dakota": "sd", "new jersey": "nj",
    "new hampshire": "nh", "new mexico": "nm", "rhode island": "ri",
    "west virginia": "wv", "district of columbia": "dc",
}

IN_STATES = {
    "andhra pradesh": "ap", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "odisha": "od",
    "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "ts", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "pondicherry": "py", "puducherry": "py",
    "andaman and nicobar islands": "an", "lakshadweep": "ld", "dadra and nagar haveli": "dn",
    "secunderabad": "tg", "bangalore": "ka", "bengaluru": "ka", "mysore": "ka",
    "mysuru": "ka", "pune": "mh", "ahmedabad": "gj", "surat": "gj", "jaipur": "rj",
    "lucknow": "up", "kanpur": "up", "nagpur": "mh", "indore": "mp", "bhopal": "mp",
    "patna": "br", "kolkata": "wb", "chennai": "tn", "madras": "tn", "coimbatore": "tn",
    "hyderabad": "ts", "kochi": "kl", "cochin": "kl", "thiruvananthapuram": "kl",
    "guwahati": "as", "chandigarh": "ch", "dehradun": "uk", "shimla": "hp",
    "bhubaneswar": "od", "bhubaneshwar": "od", "goa": "ga", "panaji": "ga",
}

FR_REGIONS = {
    "hauts de france": "hauts de france",
    "nouvelle aquitaine": "nouvelle aquitaine",
    "pays de la loire": "pays de la loire",
    "ile de france": "ile de france",
    "auvergne rhone alpes": "auvergne rhone alpes",
    "grand est": "grand est",
    "normandie": "normandie", "bretagne": "bretagne", "occitanie": "occitanie",
    "corse": "corse", "provence alpes cote d azur": "provence alpes cote d azur",
    "centre val de loire": "centre val de loire",
    "bourgogne franche comte": "bourgogne franche comte",
    "france": "fr",
}

FR_DISTRICTS = {
    "gironde": "gironde", "loire atlantique": "loire atlantique",
    "nord": "nord", "pas de calais": "pas de calais", "seine maritime": "seine maritime",
    "rhone": "rhone", "isere": "isere", "var": "var", "bouches du rhone": "bouches du rhone",
    "haute garonne": "haute garonne", "maine et loire": "maine et loire",
    "finistere": "finistere", "herault": "herault", "gard": "gard",
    "loire": "loire", "charente maritime": "charente maritime",
}

# tokens that can be either a US state abbrev or a street/other word.
# values are the canonical 2-letter code, so "MD" and "Maryland" collapse together.
AMBIGUOUS_US = {
    "al": "al", "ak": "ak", "ar": "ar", "az": "az", "ca": "ca", "co": "co",
    "ct": "ct", "de": "de", "fl": "fl", "ga": "ga", "hi": "hi", "id": "id",
    "il": "il", "in": "in", "ia": "ia", "ks": "ks", "ky": "ky", "la": "la",
    "me": "me", "md": "md", "ma": "ma", "mi": "mi", "mn": "mn", "ms": "ms",
    "mo": "mo", "mt": "mt", "ne": "ne", "nv": "nv", "nh": "nh", "nj": "nj",
    "nm": "nm", "ny": "ny", "nc": "nc", "nd": "nd", "oh": "oh", "ok": "ok",
    "or": "or", "pa": "pa", "ri": "ri", "sc": "sc", "sd": "sd", "tn": "tn",
    "tx": "tx", "ut": "ut", "vt": "vt", "va": "va", "wa": "wa", "wv": "wv",
    "wi": "wi", "wy": "wy", "dc": "dc",
}

# region tables selected by the record's country label
REGION_TABLES = {
    "us": (US_STATES, AMBIGUOUS_US, None),
    "india": (IN_STATES, None, None),
    "france": (FR_REGIONS, None, FR_DISTRICTS),
}

MAX_REGION_WORDS = 4


def _region_lookup(country: str):
    states, ambig, extra = REGION_TABLES.get(country, (US_STATES, AMBIGUOUS_US, None))
    table = dict(states)
    if extra:
        table.update(extra)
    return table, ambig

# --------------------------------------------------------------------------
# name canonicalization tables
# --------------------------------------------------------------------------

# legal-form / boilerplate tokens dropped from the *_core view
LEGAL_FORMS = {
    # english / indian / us
    "llc", "l l c", "lc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llp", "lp", "plc", "pllc", "pc", "pa", "pvt", "pvtltd",
    "private", "prv", "sd", "sp", "sdn", "bhd", "sendirian", "berhad",
    # french
    "sarl", "sas", "sasu", "sasn", "eurl", "snc", "sci", "scs", "sca", "scp",
    "sa", "sas", "ei", "eirl", "sirl", "snc", "sau", "sao", "sas",
    # german / dutch / nordic / others
    "gmbh", "mbh", "ag", "kg", "kgaa", "ohg", "gbr", "bv", "nv", "cv", "vof",
    "oy", "oyj", "ab", "asa", "aps", "as", "kk", "kft", "spa", "srl", "sapa",
    "pty", "pte", "inc", "sdn", "bhd",
    # boilerplate
    "the", "and", "of", "trading", "as", "dba", "t", "a", "an",
}

# multi-word legal forms removed as a unit
LEGAL_PHRASES = [
    "pvt ltd", "pvt ltd", "private limited", "co ltd", "company limited",
    "and company", "and co", "and sons", "and associates", "and partners",
]

_NOISE_PREFIX = re.compile(r"^(mr|mrs|ms|miss|dr|prof|sri|shri|mme|mlle|m|mme)\s+")
_DBA = re.compile(r"\b(trading as|traded as|doing business as|d/b/a|dba)\b")


def _drop_legal(tokens):
    out = []
    for t in tokens:
        if t in LEGAL_FORMS:
            continue
        out.append(t)
    return out


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def norm_name(raw: str):
    """Return ``(name_norm, name_core, nonlatin)`` for a raw business_name."""
    nonlatin = has_non_latin(raw)
    s = _DEGREE.sub(" no ", raw)
    s = squash(fold_symbols(to_latin(s)))
    if not s:
        return "", "", nonlatin
    toks = s.split()
    core = _drop_legal(toks)
    if not core:
        core = toks
    # de-duplicate while preserving order ("Private Rkk Products" -> 2 tokens)
    seen = set()
    ded = []
    for t in core:
        if t not in seen:
            seen.add(t)
            ded.append(t)
    if not ded:
        ded = toks
    core_s = " ".join(ded)
    if nonlatin:
        core_s = collapse_vowels(core_s)
    return " ".join(toks), core_s, nonlatin


def norm_addr(raw: str, country: str = ""):
    """Return ``(addr_norm, addr_core, nonlatin)`` for a raw business_address."""
    nonlatin = has_non_latin(raw)
    s = _DEGREE.sub(" no ", raw)
    s = squash(fold_symbols(to_latin(s)))
    if not s:
        return "", "", nonlatin
    toks = s.split()
    if nonlatin:
        toks = collapse_vowels(" ".join(toks)).split()

    ckey = (country or "").strip().lower()
    region, ambig = _region_lookup(ckey)
    out = []
    i = 0
    n = len(toks)
    while i < n:
        # greedy longest-match for multi-word regions / states ("new york", "ile de france")
        hit = None
        for w in range(min(MAX_REGION_WORDS, n - i), 0, -1):
            key = " ".join(toks[i:i + w])
            if key in region:
                hit = (key, w)
                break
        if hit:
            out.append(region[hit[0]])
            i += hit[1]
            continue

        t = toks[i]
        prev_num = i > 0 and toks[i - 1].isdigit()
        next_num = i + 1 < n and toks[i + 1].isdigit()
        if t == "n" and next_num:
            out.append(UNIT_ABBR["no"])          # "N 32" / "N deg 32" -> house number
        elif ambig and t in ambig and not prev_num:
            out.append(ambig[t])
        elif t in STREET_ABBR:
            out.append(STREET_ABBR[t])
        elif t in UNIT_ABBR:
            out.append(UNIT_ABBR[t])
        elif t in DIRECTIONS and (i == 0 or prev_num or next_num):
            out.append(DIRECTIONS[t])
        else:
            out.append(t)
        i += 1

    # a leading "n" before digits is the french/indian house-number marker
    if out and out[0] == "n":
        out[0] = "unit"
    norm = " ".join(out)

    # drop pure-noise tokens for the core view
    core = [t for t in out if t not in ("unit", "the", "of", "and", "de", "du", "des", "le", "la")]
    if not core:
        core = out
    return norm, " ".join(core), nonlatin


def name_tokens(core: str):
    return core.split() if core else ()


def addr_numbers(norm: str):
    """Numeric-ish tokens of an address: house numbers, blocks, postcodes, floors."""
    out = []
    for t in norm.split():
        if t.isdigit():
            out.append(t)
        elif t in ("unit",) :
            continue
    return out


def country_code(c: str, table: dict) -> int:
    return table.get(c.strip().lower(), 0)
