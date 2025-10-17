import hashlib

def hashstring(data: str) -> bytes:
    # 1将字符串编码为 UTF-8 字节串
    x = data.encode('utf-8')

    #  创建 SHA-256 哈希对象并计算哈希
    hasher = hashlib.sha256()
    hasher.update(x)
    s = hasher.digest()

    return s