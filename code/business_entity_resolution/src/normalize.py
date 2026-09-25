import re
import unicodedata

LEGAL_SUFFIXES = [
    # English
    'incorporated', 'inc', 'corporation', 'corp', 'company', 'co',
    'limited', 'ltd', 'llc', 'llp', 'lp', 'plc', 'pvt', 'private',
    'public', 'holdings', 'holding', 'group', 'grp', 'enterprises',
    'enterprise', 'intl', 'international', 'industries', 'ind',
    'solutions', 'services', 'technologies', 'technology', 'tech',
    'associates', 'assoc', 'partners', 'partnership',
    # French
    'sarl', 'sas', 'sa', 'eurl', 'sasu', 'sci', 'snc', 'gie',
    'societe', 'société', 'ste',
    # Indian
    'pvt ltd', 'private limited',
]
LEGAL_SUFFIXES.sort(key=len, reverse=True)

LEGAL_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in LEGAL_SUFFIXES) + r')\.?\b',
    re.IGNORECASE
)

ABBREVIATIONS = {
    'st': 'street', 'rd': 'road', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
    'sq': 'square', 'hwy': 'highway', 'pkwy': 'parkway',
    'cir': 'circle', 'apt': 'apartment', 'ste': 'suite',
    'bldg': 'building', 'fl': 'floor', 'dept': 'department',
    'mt': 'mount', 'ft': 'fort', 'pt': 'point',
    # French
    'av': 'avenue', 'bd': 'boulevard', 'r': 'rue',
    'pl': 'place', 'imp': 'impasse', 'ch': 'chemin',
    # Indian
    'nagar': 'nagar', 'marg': 'marg', 'nr': 'near',
}

POSTAL_RE = re.compile(r'\b(\d{5,6})\b')
HOUSE_NUM_RE = re.compile(r'^(\d+[\-/]?\d*)\s')


def strip_accents(text):
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def normalize_text(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower().strip()
    text = strip_accents(text)
    text = text.replace('&', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_legal_suffix(name):
    name_lower = normalize_text(name)
    found_suffixes = []
    for suffix in LEGAL_SUFFIXES:
        pattern = re.compile(r'\b' + re.escape(suffix) + r'\b')
        if pattern.search(name_lower):
            found_suffixes.append(suffix)
            name_lower = pattern.sub('', name_lower).strip()
    name_lower = re.sub(r'\s+', ' ', name_lower).strip()
    return name_lower, ' '.join(found_suffixes)


def expand_abbreviations(text):
    if not text:
        return text
    tokens = text.split()
    expanded = []
    for t in tokens:
        expanded.append(ABBREVIATIONS.get(t, t))
    return ' '.join(expanded)


def extract_postal_code(address):
    if not isinstance(address, str):
        return '', ''
    match = POSTAL_RE.search(address)
    if match:
        return match.group(1), POSTAL_RE.sub('', address).strip()
    return '', address


def extract_house_number(address):
    if not isinstance(address, str):
        return '', ''
    match = HOUSE_NUM_RE.match(address)
    if match:
        return match.group(1), address[match.end():].strip()
    return '', address


def make_acronym(name):
    tokens = name.split()
    if len(tokens) < 2:
        return ''
    return ''.join(t[0] for t in tokens if t)


def normalize_record(row):
    name_raw = row.get('business_name', '')
    addr_raw = row.get('business_address', '')
    country = row.get('country', '')

    name_clean = normalize_text(name_raw)
    name_no_suffix, legal_suffix = extract_legal_suffix(name_raw)

    addr_clean = normalize_text(addr_raw)
    postal, addr_no_postal = extract_postal_code(addr_clean)
    house_num, addr_no_house = extract_house_number(addr_no_postal)

    name_expanded = expand_abbreviations(name_no_suffix)
    addr_expanded = expand_abbreviations(addr_no_house)

    acronym = make_acronym(name_no_suffix)

    return {
        'name_clean': name_clean,
        'name_no_suffix': name_no_suffix,
        'name_expanded': name_expanded,
        'legal_suffix': legal_suffix,
        'addr_clean': addr_clean,
        'addr_expanded': addr_expanded,
        'postal_code': postal,
        'house_number': house_num,
        'acronym': acronym,
        'country': normalize_text(country),
    }


def normalize_dataframe(df):
    import pandas as pd
    records = df.apply(normalize_record, axis=1, result_type='expand')
    return pd.concat([df[['entity_id']], records], axis=1)
