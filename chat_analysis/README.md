# Chat Sentiment Analysis

Performs a live sentiment analysis of the twitch chat emotes to determine 
that chat's sentiment on the viability of run. 

## Method

- Create a human defined dictionary of relevant emotes and phrases with assigned
valence scores 
- At a time $t$ parse through all the relevant tokens of the chat over the 
past $t'$ seconds and take an exponentially weighted moving average of the 
valence