import re
from collections import Counter
from utils import read_input
from datetime import datetime, timedelta


VALENCE_DICT = {
    "omegaLUL": -1,
    "forsenE": 1
}


def get_top_3_tokens(message):
    keywords_pattern = "|".join(re.escape(s) for s in VALENCE_DICT.keys())
    matches = re.findall(keywords_pattern, message)
    counts = Counter(matches)
    top_3 = [token for token, count in counts.most_common(3)]
    print(top_3)
    return tuple(top_3)

def get_user_valence(top_3_tokens):
    total_sum = 0
    for token in top_3_tokens:
        total_sum += VALENCE_DICT[token]
    return total_sum


def chat_analysis(chat_df, time, time_delta):
    """Finds the average of the valence of the chat in the past `time_delta` 
    ending at `time` in the offline setting

    Args:
        chat_df (dataframe): The entire history of the chat
        time (datetime): The right time anchor of the average
        time_delta (timedelta): The size of the time window to take the average
        over

    Returns:
        float: The average valence from time to time - timedelta
    """
    
    time_filtered_df = chat_df[
        (chat_df['Time'] >= (time - time_delta)) &
        (chat_df['Time'] <= (time))
        ]
    print(time_filtered_df)
    by_user_time_df = time_filtered_df.groupby('User')['Message'].sum()
    user_top_tokens = by_user_time_df.apply(get_top_3_tokens)
    valence_scores = user_top_tokens.apply(get_user_valence)
    average_valence = valence_scores.sum() / len(user_top_tokens)
    return average_valence


chat_df = read_input("./example.txt")
time_str = "3:28:46"
t = datetime.strptime(time_str, "%H:%M:%S")
dt = timedelta(hours=1)

print(chat_analysis(chat_df, t, dt))


