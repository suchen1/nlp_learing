
def get_ngram(input_text,n_gram):
    res = set(zip(*[input_text[i:] for i in range(n_gram)]))
    return res

if __name__ == "__main__":
    input_text = "我爱自然语言处理"
    n_gram = 2
    result = list(get_ngram(input_text, n_gram)) +list(get_ngram(input_text, 1))
    print(f"{n_gram}-gram结果:", result)