import numpy as np
import keras


def counter(text: str) -> np.array:
    """Count the number of each alphabet characters present in the text.

    Args:
        text (str): Text

    Returns:
        np.array: Number of each alphabet characters present in the text
    """

    alphabet = "abcdefghijklmnopqrstuvwxyzàèéùüöäïçîûôóãâá"
    counter = list()

    for i in range(len(alphabet)):
        count = [lst for lst, v in enumerate(text) if v.lower() == alphabet[i]]
        counter.append(len(count)/len(text))
    counter = np.array(counter)

    return counter


model = keras.saving.load_model("LanguageDetection.keras")
languages = [
    "English",
    "Portuguese",
    "French",
    "Dutch",
    "Spanish",
    "Modern Greek",
    "Italian",
    "Turkish",
    "Germany",
    "Polish"
]

path = input("Path to the file containing your text : ")

with open(path, "r") as f:
    text = f.read()

text = counter(text).reshape(1, 42)

print(
    "Your text is written in",
    languages[np.argmax(model.predict(text))],
    "!"
)
