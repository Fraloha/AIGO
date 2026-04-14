import torch

with open('', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
vocabulary_size = len(chars)

int_values = { character:i for i,character in enumerate(chars) }
string_values = { i:character for i,character in enumerate(chars) }

encode = lambda string: [int_values[current] for current in string]
decode = lambda list_of_int: [string_values[current] for current in list_of_int]

encoded_hello = torch.tensor(encode('hello'), dtype = torch.long)
decoded_hello = decode(encoded_hello) 

