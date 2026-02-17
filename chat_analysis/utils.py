from datetime import datetime
import re
import pandas as pd 

def read_input(chat_file):
    """Read the chat file line by line

    Args:
        chat_file (string): path to chat html file
    """
    chat = []
    with open(chat_file,'r') as f:
        for line in f:

            time_stamp_match = re.search(r'\[(.*?)\]', line)
            if time_stamp_match:
                time_stamp = time_stamp_match.group(1)
                time_stamp_datetime = datetime.strptime(time_stamp, "%H:%M:%S")
            else:
                raise ValueError("Couldn't find time stamp. time_stamp_match is false")
            
            user_name_match = re.search(r'\[(.*?)\]', line)
            if user_name_match:
                user_name = user_name_match.group(1).strip()
            else:
                raise ValueError("Couldn't find user name. user_name_match is false")
            
            message = line.split(':', 3)[3].strip()
            chat.append([time_stamp_datetime, user_name, message])
    return pd.DataFrame(chat, columns=['Time', 'User', 'Message'])
    
        
            
                


