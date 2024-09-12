import numpy as np
import pandas as pd
import keras 

def counter(x):

    alphabet = "abcdefghijklmnopqrstuvwxyzàèéùüöäïçîûôóãâá"
    counter = list()

    for i in range(len(alphabet)):
        count = [lst for lst, v in enumerate(x) if v.lower() == alphabet[i]]
        counter.append(len(count)/len(x))
    counter = np.array(counter)
    
    return counter

model = keras.saving.load_model("LanguageDetection.keras")
languages = ['English', 'Portuguese', 'French', 'Dutch', 'Spanish', 'Modern Greek', 'Italian', 'Turkish', 'Germany', 'Polish']

path = input("Path to the file containing your text : ")

with open(path, 'r') as f:
    text = f.read()

text = counter(text).reshape(1,42)

print("Your text is written in",languages[np.argmax(model.predict(text))],"!") 
