def convert_bsf(data, bsf_markup, converter):
    def join_simple_chunk(tokens, tags):
        return ' '.join(f"{token}/{tag}" for token, tag in zip(tokens, tags))

    # Parse the BSF markup
    annotations = []
    for line in bsf_markup.strip().split('\n'):
        parts = line.split()
        if len(parts) < 4:
            continue
        entity_type = parts[1]
        start = int(parts[2])
        end = int(parts[3])
        annotations.append((start, end, entity_type))

    # Tokenize the data
    tokens = data.split()
    token_starts = []
    current_pos = 0
    for token in tokens:
        token_starts.append(current_pos)
        current_pos += len(token) + 1  # +1 for the space

    # Initialize tags
    tags = ['O'] * len(tokens)

    # Apply annotations
    for start, end, entity_type in annotations:
        # Find the start and end token indices
        start_token_idx = next((i for i, pos in enumerate(token_starts) if pos == start), None)
        end_token_idx = next((i for i, pos in enumerate(token_starts) if pos + len(tokens[i]) == end), None)

        if start_token_idx is None or end_token_idx is None:
            continue

        if converter.lower() == 'iob':
            tags[start_token_idx] = f'B-{entity_type}'
            for i in range(start_token_idx + 1, end_token_idx + 1):
                tags[i] = f'I-{entity_type}'
        elif converter.lower() == 'beios':
            if start_token_idx == end_token_idx:
                tags[start_token_idx] = f'S-{entity_type}'
            else:
                tags[start_token_idx] = f'B-{entity_type}'
                for i in range(start_token_idx + 1, end_token_idx):
                    tags[i] = f'I-{entity_type}'
                tags[end_token_idx] = f'E-{entity_type}'
        else:
            raise ValueError("Converter must be either 'beios' or 'iob'")

    # Join tokens with their tags
    return join_simple_chunk(tokens, tags)

