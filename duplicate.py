with open('naive_output_probs.txt', 'r', encoding='utf-8', errors='ignore') as file:
    word_tag_pairs = set()
    for line in file:
        # Strip whitespace characters from the beginning and end of the line
        line = line.strip()
        # Check if the line is empty or starts with a semicolon character
        if not line or line.startswith(';'):
            continue
        try:
            # Extract the word and tag from the line
            fields = line.split()
            word = fields[0]
            tag = fields[1]
            # Check if the word-tag pair has already been seen
            if (word, tag) in word_tag_pairs:
                print("Duplicate word-tag pair found:", (word, tag))
            else:
                word_tag_pairs.add((word, tag))
        except (IndexError, ValueError):
            # If extracting the tag fails, skip the line and continue to the next one
            continue
