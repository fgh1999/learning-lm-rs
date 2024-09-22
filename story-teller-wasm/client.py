import requests
import tokenizers

srv = "http://localhost:10022"
tokenizer_path = "../models/story/tokenizer.json"
# tokenizer_path = "tokenizer.json"
tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)


def hello():
    return requests.get(f"{srv}").text

def story():
    generated_token_ids = requests.get(f"{srv}/story").json()
    return tokenizer.decode(generated_token_ids)

if __name__ == "__main__":
    print(hello())
    print(f"Once upon a time, {story()}")
