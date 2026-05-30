from src.vocab.base import BaseTokenizer
import os


def load_file(file_path="", encoding = "utf-8"):
    assert os.path.exists(file_path), f"The file_path {file_path} should exists"
    with open(file_path,"r", encoding=encoding) as fin:
        lines = fin.readlines()
    lines = [l.strip() for l in lines]
    return lines