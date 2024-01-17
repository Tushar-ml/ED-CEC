import string

def clean_text(text: str) -> str:
    punctuations = string.punctuation
    result = ''

    for char in text:
        if char not in punctuations:
            result += char.lower()
    result = result.replace("  ", " ")
    return result
