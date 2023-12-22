

sentences = (
    "Jane wants to go to Shenzhen",
    "Bob wants to go to Shanghai"
)

## one hot encode
## bag of words
words = [ word for sentence in sentences for word in sentence.split(" ")]
words = tuple(set(words))
print(words)

def bag_of_words(sentence: str, words: tuple):
    return [sentence.split(" ").count(word) for word in words]

print(bag_of_words(sentences[0], words))
print(bag_of_words(sentences[1], words))

tokens = [ sentence.split(" ") for sentence in sentences]
def vec
