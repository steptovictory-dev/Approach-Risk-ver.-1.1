"""Проверка исторических SHA256 с учетом только окончаний строк текстовых файлов"""
import hashlib


def matches_sha256(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected:
        return True
    if path.suffix.lower() not in {'.py', '.json', '.csv', '.md', '.txt'}:
        return False
    raw.decode('utf-8')
    lf = raw.replace(b'\r\n', b'\n')
    return any(hashlib.sha256(value).hexdigest() == expected
               for value in (lf, lf.replace(b'\n', b'\r\n')))
