from transformers import BertModel, BertTokenizer
import torch
from modules.utils import clean_text

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

def wordpiece_tokenizer(text):
    text = clean_text(text)
    words = []
    for idx, word in enumerate(text.split(" ")):
        words.append(f'[UNK]')
        words.append(word)
    
    words.append("[UNK]")
    words = ' '.join(words)
    tokens = tokenizer.tokenize(words)

    return tokens

def longest_common_subsequence(list1, list2):
    m, n = len(list1), len(list2)

    # Initialize a 2D table to store the lengths of LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Build the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct the LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if list1[i - 1] == list2[j - 1]:
            lcs.append(list1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs[::-1]  # Reverse the list to get the actual LCS

def align_lists(list1, list2):
    lcs = longest_common_subsequence(list1, list2)
    aligned_list1 = []
    aligned_list2 = []

    i, j = 0, 0

    for word in lcs:
        while i < len(list1) and list1[i] != word:
            aligned_list1.append('[UNK]')
            aligned_list2.append(list1[i])
            i += 1

        while j < len(list2) and list2[j] != word:
            aligned_list1.append(list2[j])
            aligned_list2.append('[UNK]')
            j += 1

        aligned_list1.append(word)
        aligned_list2.append(word)

        i += 1
        j += 1

    # Extend lists from both sides with [UNK]
    while i < len(list1):
        aligned_list1.append('[UNK]')
        aligned_list2.append(list1[i])
        i += 1

    while j < len(list2):
        aligned_list1.append(list2[j])
        aligned_list2.append('[UNK]')
        j += 1

    return aligned_list1, aligned_list2

def assign_labels(list1, list2):
    labels = []

    for token1, token2 in zip(list1, list2):
        if token1 == "[UNK]" and token2 == "[UNK]":
            labels.append("D")
        elif token1 == token2:
            labels.append("K")
        elif token1 == "[UNK]" and token2 != "[UNK]":
            labels.append("C")
        elif token1 != "[UNK]" and token2 == "[UNK]":
            labels.append("D")
        else:
            labels.append("C")

    return labels

def get_bert_embeddings(tokens):
    # Tokenize input sequence and get BERT embeddings
    inputs = tokenizer(tokens, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling over tokens
    return embeddings

def combine_labels(prediction_list, labels):

    assert len(prediction_list)==len(labels)

    clabel = [labels[0]]
    interim_pred = [prediction_list[0]]

    for pred, lab in zip(prediction_list[1:], labels[1:]):

        if pred == '[UNK]' and pred == interim_pred[-1]:
            if clabel[-1] == "D" and lab == "C":
                clabel[-1] = lab

        else:
            interim_pred.append(pred)
            clabel.append(lab)

    return interim_pred, clabel