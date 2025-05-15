def convert_bsf(data: str, bsf_markup: str, converter: str = 'beios') -> str:
    """
    Convert data file with NER markup in Brat Standoff Format to BEIOS or IOB format.

    :param converter: iob or beios converter to use for document
    :param data: tokenized data to be converted. Each token separated with a space
    :param bsf_markup: Brat Standoff Format markup
    :return: data in BEIOS or IOB format https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)
    """

    def join_simple_chunk(chunk: str) -> list:
        if len(chunk.strip()) == 0:
            return []
        # keep the newlines, but discard the non-newline whitespace
        tokens = re.split(r'(\n)|\s', chunk.strip())
        # the re will return None for splits which were not caught in a group
        tokens = [x for x in tokens if x is not None]
        return [token + ' O' if len(token.strip()) > 0 else token for token in tokens]

    converters = {'beios': format_token_as_beios, 'iob': format_token_as_iob}
    res = []
    markup = parse_bsf(bsf_markup)

    prev_idx = 0
    m_ln: BsfInfo
    for m_ln in markup:
        res += join_simple_chunk(data[prev_idx:m_ln.start_idx])

        convert_f = converters[converter]
        res.extend(convert_f(m_ln.token, m_ln.tag))
        prev_idx = m_ln.end_idx

    if prev_idx < len(data) - 1:
        res += join_simple_chunk(data[prev_idx:])

    return '\n'.join(res)