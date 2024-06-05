# preprocessing for ml-100k
def ml100k_preprocess(text: str) -> str:
    if text.endswith(', The'):
        text = 'The ' + text[:-5]
    elif text.endswith(', A'):
        text = 'A ' + text[:-3]
    return text