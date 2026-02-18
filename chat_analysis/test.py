from utils import read_input
from datetime import timedelta, datetime
from chat_analysis import chat_analysis
import matplotlib.pyplot as plt

def main():
    df = read_input("./example.txt")
    X = []
    Y = []
    current_time = datetime.strptime("1:23:47", "%H:%M:%S")
    dt = timedelta(seconds=30)
    for i in range(0,480):
        X.append(current_time)
        Y.append(chat_analysis(df, current_time, dt))
        current_time += dt 
    plt.plot(X,Y)
    plt.show()

if __name__ == "__main__":
    main()
