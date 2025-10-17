from algorithm.AesCBCalgorithm import genKey, AesEncrypt, AesDecrypt
from algorithm.RSAalgorithm import *
from algorithm.photoHash import *
import pickle

if __name__ == "__main__":
    Usersk, Userpk = generateKeys() #生成用户的公私钥对
    Modelsk, Modelpk = generateKeys() #生成模型的公私钥对

    #用户加密流程
    with open("test_image.png", 'rb') as f:
        flag_pic = f.read()#写入图片
    key = genKey()#生成一次一密密钥
    enc_pic = AesEncrypt(flag_pic, key)#对图片进行AES加密，加密模式为CBC
    Digest = hashstring(enc_pic) #生成加密后的哈希值
    signature=sign_message(Usersk,Digest) #使用用户私钥对生成后的哈希值进行签名
    enckey=RsaEncrypt(key,Modelpk) #使用模型的公钥对key进行加密

    Message={"enc_pic":enc_pic,"signature":signature}
    message=pickle.dumps(Message)
    MES=pickle.dumps([message,enckey.decode('utf-8')]) #将加密后的图片，签名，加密后的私钥打包

    #模型解密流程
    (message, enckey) = pickle.loads(MES)
    key=RsaDecrypt(enckey,Modelsk).decode('utf-8') #使用模型私钥对key进行解密
    print(key)
    Message=pickle.loads(message)
    enccontent=Message["enc_pic"]
    digest1=hashstring(enccontent) #还原加密后图片的哈希值
    decrypted_image_data = AesDecrypt(enccontent, key) #解密图片

    # 保存解密后的图片
    with open('decrypted_image.jpg', 'wb') as f:
        f.write(decrypted_image_data)
    signature=Message["signature"]
    print(verify_signature(Userpk,digest1,signature)) #验证签名，证明图片是来源于用户









