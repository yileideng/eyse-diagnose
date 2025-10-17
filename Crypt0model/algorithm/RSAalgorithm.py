from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding,rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import hashlib
import base64

#生成rsa的公私钥对，转换为pem格式
def generateKeys():

    #生成私钥对象
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    # 从私钥对象派生公钥对象
    public_key = private_key.public_key()

    # 将私钥对象序列化为 PEM 格式
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()  # 不加密私钥文件
    ).decode('utf-8')

    # 将公钥对象序列化为 PEM 格式
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode('utf-8')

    return private_pem, public_pem

#签名算法
def sign_message(private_pem: str, message: bytes) -> bytes:
    #  从 PEM 字符串加载私钥对象
    private_key = serialization.load_pem_private_key(
        private_pem.encode('utf-8'),
        password=None ,# 如果私钥有密码，在此处提供
        backend = default_backend()
    )

    #  使用加载的私钥对象进行签名
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return signature #返回bytes

#验证签名算法
def verify_signature(public_pem: str, message: bytes, signature: bytes) -> bool:
    try:
        # 从 PEM 字符串加载公钥对象
        public_key = serialization.load_pem_public_key(
            public_pem.encode('utf-8'),
            backend = default_backend()
        )

        # 使用加载的公钥对象进行验证
        public_key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except:
        return False #如果签名有效则返回 True，否则返回 False。


def RsaEncrypt(message, public_key_pem):
    # 确保密钥是字节格式
    if isinstance(public_key_pem, str):
        public_key_pem = public_key_pem.encode('utf-8')

    # 加载公钥
    public_key = serialization.load_pem_public_key(
        public_key_pem,
        backend=default_backend()
    )

    # 确保消息是字节类型
    if isinstance(message, str):
        message = message.encode('utf-8')

    # 使用PKCS1v15填充进行加密
    cipher_text = public_key.encrypt(
        message,
        padding.PKCS1v15()
    )

    # Base64编码并打印结果
    ciphertext = base64.b64encode(cipher_text)
    print(ciphertext.decode('utf-8'))

    return ciphertext


def RsaDecrypt(encryptText, private_key_pem):
    # 确保密钥是字节格式
    if isinstance(private_key_pem, str):
        private_key_pem = private_key_pem.encode('utf-8')

    # 加载私钥
    private_key = serialization.load_pem_private_key(
        private_key_pem,
        password=None,
        backend=default_backend()
    )

    # 处理加密文本
    if isinstance(encryptText, str):
        encrypt_text = base64.b64decode(encryptText)
    elif isinstance(encryptText, bytes):
        # 检查是否已经是原始密文（非Base64编码）
        # 这里我们假设如果长度不等于密钥大小/8（2048位密钥的典型密文长度是256字节），则需要解码
        if len(encryptText) != private_key.key_size // 8:
            try:
                encrypt_text = base64.b64decode(encryptText)
            except:
                # 如果Base64解码失败，则保持原样
                pass

    # 使用PKCS1v15填充进行解密
    text = private_key.decrypt(
        encrypt_text,
        padding.PKCS1v15()
    )

    print("测试点test", type(text))
    return text


if __name__ == "__main__":
    #  生成密钥对
    private_key_pem, public_key_pem = generateKeys()
    # print("\n--- 私钥 ---")
    # print(private_key_pem)
    # print("\n--- 公钥 ---")
    # print(public_key_pem)

    # 定义原始消息
    original_message_str = "Hello, Crypto"
    original_message_bytes = original_message_str.encode('utf-8')
    print(f"\n原始消息: '{original_message_str}'")

    #  签名与验证测试
    signature = sign_message(private_key_pem, original_message_bytes)

    is_valid = verify_signature(public_key_pem, original_message_bytes, signature)
    print(f"验证结果: {'成功' if is_valid else '失败'}")

    #用错误信息签名
    invalid_message_bytes = b"this is not the correct message"
    is_valid_fail = verify_signature(public_key_pem, invalid_message_bytes, signature)
    print(f"验证结果: {'成功' if is_valid_fail else '失败'}")

    # 4. 加密与解密测试
    encrypted_message = RsaEncrypt(original_message_bytes, public_key_pem)
    print(f"加密后的消息: {encrypted_message.decode('utf-8')}")

    decrypted_message_bytes = RsaDecrypt(encrypted_message, private_key_pem)
    decrypted_message_str = decrypted_message_bytes.decode('utf-8')
    print(f"解密后的消息: '{decrypted_message_str}'")

    if original_message_str == decrypted_message_str:
        print("验证成功")
    else:
        print("验证失败")
