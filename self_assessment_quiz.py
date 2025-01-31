# code for the self assessment quiz

import re

def question_six(input):
    words = re.findall(r'\b\w+\b', input.lower())
    dict = {}
    ind = []

    i = 0
    for word in words:
        if word in dict:
            ind.append(dict.get(word))
        else:
            dict.update({word: i})
            ind.append(i)
            i += 1

    print(dict)
    print(ind)

def question_seven():
    print(*range(2,101,2))


if __name__ == '__main__':
    question_seven()