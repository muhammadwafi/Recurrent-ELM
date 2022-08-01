import email

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from bs4 import BeautifulSoup
import glob

ps = PorterStemmer()
CLEAN = re.compile('[^a-zA-Z]')


def case_folding(data):
    return data.lower()


def stopwords_removal_and_stemming(data):
    data_arr = data.split()
    result = list()
    for word in data_arr:
        temp = word
        if len(temp) == 2:
            temp = remove_duplicate(word)
        if not temp in stopwords.words('english') and len(temp) > 1:
            result.append(ps.stem(temp))
    return result


def remove_duplicate(str):
    s = set(str)
    return "".join(s)


def cleaning(data):
    return re.sub(CLEAN, ' ', data)


def remove_html_tag(data):
    return BeautifulSoup(data, "lxml").text


def get_file_names(path, filetype, n=10):
    if filetype == None:
        path += "/*"
    else:
        path += "/*." + filetype
    file_list = glob.glob(path)
    return file_list[:10]


def open_email(paths):
    emails = list()
    for path in paths:
        print("opening", path)
        message = email.message_from_file(open(path, encoding="cp1252"))
        payload = message.get_payload()
        body = get_body_email(payload)
        emails.append(body)
    return emails


def get_body_email(data):
    if type(data) == str:
        return data
    return get_body_email(data[0].get_payload())


def get_fitur(datas):
    fitur = []
    for data in datas:
        fitur += data
    return list(set(fitur))
