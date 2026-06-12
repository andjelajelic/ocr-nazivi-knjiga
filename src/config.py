# src/config.py

# Dimenzije na koje ćemo uniformno smanjiti sve slike
IMG_WIDTH = 256
IMG_HEIGHT = 64

# Rečnik koji razlikuje velika i mala slova + naša slova + specijalni znakovi
# VAŽNO: Prvi karakter je '-' (blank) i on je obavezan za CTC loss!
CHARACTERS = [
    '-', ' ', 
    'A', 'B', 'C', 'Č', 'Ć', 'D', 'Đ', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'Š', 'T', 'U', 'V', 'Z', 'Ž',
    'a', 'b', 'c', 'č', 'ć', 'd', 'đ', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 'š', 't', 'u', 'v', 'z', 'ž',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '.', ',', ':', ';',
    '!', '?', '"', "'", '(', ')', '&', '+', '/',
]

NUM_CLASSES = len(CHARACTERS)

# Automatsko generisanje indeksa za karaktere
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS)}