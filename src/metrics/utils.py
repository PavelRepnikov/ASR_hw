import numpy as np

def calc_cer(target_text: str, predicted_text: str) -> float:
    if target_text == '':
        return 0.0  # If the target text is empty, the CER is 0 (no error to calculate)
    
    # Levenshtein distance calculation for CER
    target_len = len(target_text)
    predicted_len = len(predicted_text)

    # Create a matrix for dynamic programming
    dp = np.zeros((target_len + 1, predicted_len + 1))

    # Initialize the first row and column
    for i in range(target_len + 1):
        dp[i][0] = i
    for j in range(predicted_len + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, target_len + 1):
        for j in range(1, predicted_len + 1):
            cost = 0 if target_text[i - 1] == predicted_text[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,        # Deletion
                           dp[i][j - 1] + 1,        # Insertion
                           dp[i - 1][j - 1] + cost) # Substitution

    # CER is the edit distance divided by the length of the target text
    cer = dp[target_len][predicted_len] / target_len
    return cer


def calc_wer(target_text: str, predicted_text: str) -> float:
    if not target_text:
        return 1.1 if predicted_text else 0.0  # If the target text is empty, the WER is 0 (no error to calculate)
    
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    target_len = len(target_words)
    predicted_len = len(predicted_words)

    # Create a matrix for dynamic programming
    dp = np.zeros((target_len + 1, predicted_len + 1))

    # Initialize the first row and column
    for i in range(target_len + 1):
        dp[i][0] = i
    for j in range(predicted_len + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, target_len + 1):
        for j in range(1, predicted_len + 1):
            cost = 0 if target_words[i - 1] == predicted_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,        # Deletion
                           dp[i][j - 1] + 1,        # Insertion
                           dp[i - 1][j - 1] + cost) # Substitution

    # WER is the edit distance divided by the number of words in the target
    wer = dp[target_len][predicted_len] / target_len
    return wer

