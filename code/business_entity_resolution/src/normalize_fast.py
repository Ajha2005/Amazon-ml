"""
Vectorized normalization of business names and addresses.

Rules are language-agnostic (applied identically to every record, whatever its
country) so that the spelling variants listed in the challenge collapse to one
canonical form on both sides of a pair: legal forms (Pvt Ltd / Private Limited,
S.A.R.L. / SARL, M/s prefix), joining words (& / and / et), transliterations
(Shri / Sri / Shree), street-type and directional abbreviations
(St / Street, Bd / Boulevard, N / North, Ngr / Nagar) and French elisions (l', d').
"""
import re

import pandas as pd

LEGAL_SUFFIXES = [
    'incorporated', 'corporation', 'company', 'limited', 'private limited',
    'pvt ltd', 'llc', 'llp', 'pllc', 'plc', 'pvt', 'inc', 'corp', 'ltd', 'co', 'lp',
    'holdings', 'holding', 'group', 'enterprises', 'enterprise',
    'international', 'industries', 'solutions', 'services',
    'technologies', 'technology', 'associates', 'partners',
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'selarl', 'scop', 'sa',
    'and cie', 'cie', 'gmbh',
]
LEGAL_SUFFIXES.sort(key=len, reverse=True)
_SUFFIX_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in LEGAL_SUFFIXES) + r')\.?\s*$')
_SUFFIX_ALT = '|'.join(re.escape(x) for x in LEGAL_SUFFIXES)
# Trailing suffix words are stripped repeatedly ("abc technologies pvt ltd" and
# "abc technologies" both become "abc"); the leading \s+ keeps names non-empty.
_STRIP_SUFFIX = re.compile(r'\s+(?:' + _SUFFIX_ALT + r'|and)\s*$')

# Legal forms and courtesy prefixes written before the name ("SARL Dupont", "M/s Sharma Traders")
LEGAL_PREFIXES = ['m s', 'ms', 'messrs', 'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc',
                  'selarl', 'ste', 'societe', 'ets', 'etablissements']
LEGAL_PREFIXES.sort(key=len, reverse=True)
_PREFIX_PATTERN = re.compile(r'^(' + '|'.join(re.escape(s) for s in LEGAL_PREFIXES) + r')\s+(?=\S)')

_MULTI_SPACE = re.compile(r'\s+')
_PUNCT = re.compile(r'[^\w\s]')
_COMBINING = re.compile(r'[̀-ͯ]')
_APOSTROPHE = re.compile(r"['’`]")
_DOTTED = re.compile(r'\b(?:[a-z]\.\s?){2,}(?:[a-z]\b)?')
_POSTAL_RE = re.compile(r'\b(\d{5,6})\b')
_HOUSE_RE = re.compile(r'^(\d+[\-/]?\d*)\s')
_DIGIT_ALPHA = re.compile(r'(\d)(?!(?:st|nd|rd|th)\b)([a-z])')
_ALPHA_DIGIT = re.compile(r'([a-z])(\d)')
_NUMERO = re.compile(r'\bn\s*\u00b0')
_NO_BEFORE_DIGIT = re.compile(r'\bno\s+(?=\d)')
_LEADING_ZEROS = re.compile(r'\b0+(?=\d)')
_COUNTRY_IN_NAME = re.compile(r'\(\s*(?:france|india|usa)\s*\)')
_TRAILING_COUNTRY = re.compile(r'\s(?:france|india|usa)$')
_TRADE_NAME = re.compile(r'^.*?\b(?:trading as|doing business as|dba|aka|t a)\s+(?=\S)')
_INNER_FORMS = re.compile(r'\b(?:sarl|sasu|sas|eurl|sci|snc|selarl|scop|gmbh)\b')
_CEDEX = re.compile(r'\bcedex(?:\s+\d{1,3})?\b')
_LIGATURES = str.maketrans({'œ': 'oe', 'æ': 'ae', 'ß': 'ss', 'ø': 'o'})

# Spelled-out legal forms left after punctuation removal ("S.A.R.L" -> "s a r l")
SPACED_FORMS = {'s a r l': 'sarl', 's a s u': 'sasu', 's a s': 'sas', 'e u r l': 'eurl',
                's c i': 'sci', 's n c': 'snc', 'p v t': 'pvt', 'l l c': 'llc', 'l l p': 'llp'}

NAME_WORDS = {
    'et': 'and', 'saint': 'st', 'sainte': 'ste',
    'shri': 'sri', 'shree': 'sri', 'sree': 'sri',
    'pvt': 'private', 'ltd': 'limited', 'co': 'company', 'corp': 'corporation',
    'inc': 'incorporated', 'bros': 'brothers', 'intl': 'international',
    'mfg': 'manufacturing', 'svc': 'services', 'svcs': 'services',
    'assoc': 'associates', 'assn': 'association', 'ctr': 'center', 'centre': 'center',
    'natl': 'national', 'mgmt': 'management', 'mgt': 'management', 'hosp': 'hospital',
    'univ': 'university', 'inst': 'institute', 'engg': 'engineering',
    'dept': 'department', 'grp': 'group', 'hldgs': 'holdings',
}

ADDR_WORDS = {
    # street types
    'st': 'street', 'saint': 'street', 'rd': 'road', 'ave': 'avenue', 'av': 'avenue',
    'blvd': 'boulevard', 'bd': 'boulevard', 'bld': 'boulevard', 'boul': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'hwy': 'highway', 'pkwy': 'parkway',
    'cir': 'circle', 'trl': 'trail', 'expy': 'expressway', 'fwy': 'freeway',
    'plz': 'plaza', 'pl': 'place', 'sq': 'square', 'rte': 'route',
    'r': 'rue', 'ch': 'chemin', 'chem': 'chemin', 'imp': 'impasse', 'all': 'allee',
    'fbg': 'faubourg', 'fg': 'faubourg', 'crs': 'cours', 'res': 'residence',
    # units and buildings
    'apt': 'apartment', 'ste': 'suite', 'sainte': 'suite', 'bldg': 'building',
    'fl': 'floor', 'flr': 'floor', 'rm': 'room',
    # landmarks and localities
    'nr': 'near', 'opp': 'opposite', 'ngr': 'nagar', 'sec': 'sector', 'mkt': 'market',
    'col': 'colony', 'clny': 'colony', 'extn': 'extension', 'ext': 'extension',
    'ph': 'phase', 'dist': 'district', 'chk': 'chowk', 'hts': 'heights',
    'ctr': 'center', 'centre': 'center', 'mt': 'mount', 'pt': 'point',
    # directionals
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest',
}


# Whole comma-separated address components that are only a region, state or
# country: sources include or swap them inconsistently ("..., Hauts-de-France"
# vs "..., Nord" vs nothing). Names that double as cities are left out.
REGIONS = [
    'hauts de france', 'nouvelle aquitaine', 'pays de la loire', 'ile de france', 'grand est',
    'auvergne rhone alpes', 'provence alpes cote dazur', 'occitanie', 'bretagne', 'normandie',
    'bourgogne franche comte', 'centre val de loire', 'corse',
    'nord', 'pas de calais', 'gironde', 'loire atlantique', 'somme', 'aisne', 'oise', 'vendee',
    'maine et loire', 'sarthe', 'mayenne', 'landes', 'dordogne', 'charente', 'charente maritime',
    'pyrenees atlantiques', 'lot et garonne',
    'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh', 'goa', 'gujarat',
    'haryana', 'himachal pradesh', 'jharkhand', 'karnataka', 'kerala', 'madhya pradesh',
    'maharashtra', 'manipur', 'meghalaya', 'mizoram', 'nagaland', 'odisha', 'orissa', 'punjab',
    'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura', 'uttar pradesh', 'uttarakhand',
    'west bengal', 'jammu and kashmir', 'nct of delhi',
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware',
    'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky',
    'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi',
    'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico',
    'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island',
    'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia',
    'west virginia', 'wisconsin', 'wyoming', 'district of columbia',
    'france', 'india', 'usa', 'us', 'united states', 'united states of america',
]
_REGION_RE = re.compile(
    r'(?:^|(?<=,))\s*(?:' + '|'.join(r'[\s\-]+'.join(map(re.escape, r.split())) for r in
                                  sorted(REGIONS, key=len, reverse=True)) + r')\s*(\d{5,6})?\s*(?=,|$)')

STREET_TYPES = ['rue', 'avenue', 'boulevard', 'allee', 'cours', 'place', 'chemin', 'impasse', 'route',
                'quai', 'square', 'cite', 'residence', 'faubourg', 'passage', 'promenade', 'street',
                'road', 'lane', 'drive', 'court', 'highway', 'parkway', 'circle', 'trail', 'plaza',
                'way', 'marg', 'nagar']
# House number anywhere in the address when it is not the first token
# ("hauts de france 37 rue chanzy lille", "dunkerque 29 cours francois bart")
_HOUSE_BEFORE_STREET = re.compile(r'\b(\d+)\s+(?:(?:bis|ter|[a-z])\s+)?(?:' + '|'.join(STREET_TYPES) + r')\b')


def _word_pattern(words):
    return re.compile(r'\b(' + '|'.join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r')\b')


_SPACED_RE = _word_pattern(SPACED_FORMS)
_NAME_RE = _word_pattern(NAME_WORDS)
_ADDR_RE = _word_pattern(ADDR_WORDS)


def _pre_clean(series: pd.Series) -> pd.Series:
    s = series.fillna('').str.lower().str.translate(_LIGATURES)
    s = s.str.normalize('NFKD').str.replace(_COMBINING, '', regex=True)
    return s.str.replace(_APOSTROPHE, '', regex=True)


def _post_clean(s: pd.Series) -> pd.Series:
    s = s.str.replace(_DOTTED, lambda m: m.group(0).replace('.', '').replace(' ', '') + ' ', regex=True)
    s = s.str.replace('&', ' and ', regex=False)
    s = s.str.replace(_PUNCT, ' ', regex=True)
    return s.str.replace(_MULTI_SPACE, ' ', regex=True).str.strip()


def _squash(s: pd.Series) -> pd.Series:
    return s.str.replace(_MULTI_SPACE, ' ', regex=True).str.strip()


def normalize_names(series: pd.Series) -> pd.DataFrame:
    """Returns DataFrame with name_clean, name_no_suffix, legal_suffix, acronym."""
    s = _pre_clean(series).str.replace(_COUNTRY_IN_NAME, ' ', regex=True)
    s = _post_clean(s).str.replace(_TRAILING_COUNTRY, '', regex=True)
    s = s.str.replace(_SPACED_RE, lambda m: SPACED_FORMS[m.group(1)], regex=True)
    s = s.str.replace(_NAME_RE, lambda m: NAME_WORDS[m.group(1)], regex=True)
    name_clean = s

    legal_suffix = s.str.extract(_SUFFIX_PATTERN, expand=False).fillna('')
    # "X trading as Y" / "X dba Y": keep the trade name Y
    name_no_suffix = s.str.replace(_TRADE_NAME, '', regex=True)
    for _ in range(3):
        name_no_suffix = name_no_suffix.str.replace(_STRIP_SUFFIX, '', regex=True)

    prefix = name_no_suffix.str.extract(_PREFIX_PATTERN, expand=False).fillna('')
    has_prefix = prefix != ''
    name_no_suffix = name_no_suffix.where(~has_prefix, name_no_suffix.str.replace(_PREFIX_PATTERN, '', regex=True))
    legal_suffix = legal_suffix.where(legal_suffix != '', prefix)
    # Legal forms left in the middle ("as sarl sportive", "team sarl and fils")
    inner = _squash(name_no_suffix.str.replace(_INNER_FORMS, ' ', regex=True))
    name_no_suffix = inner.where(inner != '', name_no_suffix)

    acronym = name_no_suffix.str.split().map(
        lambda tokens: ''.join(t[0] for t in tokens if t) if isinstance(tokens, list) and len(tokens) >= 2 else ''
    )
    return pd.DataFrame({
        'name_clean': name_clean,
        'name_no_suffix': name_no_suffix,
        'legal_suffix': legal_suffix,
        'acronym': acronym,
    })


def normalize_addresses(series: pd.Series) -> pd.DataFrame:
    """Returns DataFrame with addr_clean, addr_expanded, postal_code, house_number."""
    s = _pre_clean(series).str.replace(_NUMERO, 'no ', regex=True)
    s = _post_clean(s.str.replace(_REGION_RE, r' \1', regex=True))
    s = s.str.replace(_DIGIT_ALPHA, r'\1 \2', regex=True).str.replace(_ALPHA_DIGIT, r'\1 \2', regex=True)
    s = _squash(s.str.replace(_CEDEX, ' ', regex=True))
    addr_clean = s

    postal_code = s.str.extract(_POSTAL_RE, expand=False).fillna('')
    addr_no_postal = s.str.replace(_POSTAL_RE, '', regex=True)
    addr_no_postal = _squash(addr_no_postal.str.replace(_NO_BEFORE_DIGIT, '', regex=True)
                             .str.replace(_LEADING_ZEROS, '', regex=True))
    house_number = addr_no_postal.str.extract(_HOUSE_RE, expand=False).fillna('')
    addr_expanded = _squash(addr_no_postal.str.replace(_ADDR_RE, lambda m: ADDR_WORDS[m.group(1)], regex=True))
    missing = house_number == ''
    if missing.any():
        house_number = house_number.copy()
        house_number[missing] = addr_expanded[missing].str.extract(_HOUSE_BEFORE_STREET, expand=False).fillna('')

    return pd.DataFrame({
        'addr_clean': addr_clean,
        'addr_expanded': addr_expanded,
        'postal_code': postal_code,
        'house_number': house_number,
    })


def normalize_dataframe_fast(df: pd.DataFrame) -> pd.DataFrame:
    names = normalize_names(df['business_name'])
    addrs = normalize_addresses(df['business_address'])
    country = df['country'].fillna('').str.lower().str.strip()
    return pd.concat([
        df[['entity_id']].reset_index(drop=True),
        names.reset_index(drop=True),
        addrs.reset_index(drop=True),
        country.rename('country').reset_index(drop=True),
    ], axis=1)


def normalize_in_chunks_fast(df: pd.DataFrame, chunk_size: int = 500000) -> pd.DataFrame:
    """Process in chunks to control peak memory."""
    total = len(df)
    chunks = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunks.append(normalize_dataframe_fast(df.iloc[start:end]))
        if (start // chunk_size) % 5 == 0 and start > 0:
            print(f"    Normalized {end:,}/{total:,} rows...")
    return pd.concat(chunks, ignore_index=True)
