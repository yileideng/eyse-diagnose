from flask import Flask, request, jsonify, send_file
import base64
import pickle
from algorithm.AesCBCalgorithm import AesEncrypt, genKey
from algorithm.photoHash import hashstring
from algorithm.RSAalgorithm import sign_message, RsaEncrypt

app = Flask(__name__)


@app.route('/')
def home():
    return jsonify({
        "message": "图片加密API - MES格式",
        "description": "按照用户端确切流程生成MES格式的加密数据",
        "endpoints": {
            "/encrypt-image": "加密图片并生成MES (POST)",
            "/encrypt-image-mes": "直接返回MES二进制数据 (POST)",
            "/health": "健康检查"
        },
        "usage": "POST /encrypt-image 包含图片文件、用户私钥和模型公钥"
    })


@app.route('/encrypt-image', methods=['POST'])
def encrypt_image():
    try:
        # 检查必要的参数
        if 'image' not in request.files:
            return jsonify({"error": "缺少图片文件"}), 400

        if 'user_private_key' not in request.form:
            return jsonify({"error": "缺少用户私钥"}), 400

        if 'model_public_key' not in request.form:
            return jsonify({"error": "缺少模型公钥"}), 400

        # 获取上传的文件和密钥
        image_file = request.files['image']
        user_private_key = request.form['user_private_key']
        model_public_key = request.form['model_public_key']

        # 读取图片数据
        image_data = image_file.read()
        if len(image_data) == 0:
            return jsonify({"error": "图片文件为空"}), 400

        print("开始加密流程...")

        # 生成AES密钥
        aes_key = genKey()
        print(f"生成的AES密钥: {aes_key}")

        # 对图片进行AES加密
        enc_pic = AesEncrypt(image_data, aes_key)
        print("图片AES加密完成")

        # 生成加密后图片的哈希值
        Digest = hashstring(enc_pic)
        print("图片哈希计算完成")

        #  使用用户私钥对哈希值进行签名
        signature = sign_message(user_private_key, Digest)
        print("签名完成")

        # 使用模型公钥对AES密钥进行加密
        enckey = RsaEncrypt(aes_key, model_public_key)
        print("AES密钥RSA加密完成")

        # 打包数据
        Message = {
            "enc_pic": enc_pic,  # 字符串格式的base64编码
            "signature": signature  # 字节数据
        }

        # 第一次序列化
        message = pickle.dumps(Message)

        # 第二次序列化，将enckey转换为字符串
        enckey_str = enckey.decode('utf-8')
        MES = pickle.dumps([message, enckey_str])
        print("MES打包完成")

        # 返回结果（包含MES的base64编码，便于JSON传输）
        mes_b64 = base64.b64encode(MES).decode('utf-8')

        return jsonify({
            "status": "success",
            "message": "图片加密完成，MES已生成",
            "data": {
                "mes_data": mes_b64,  # MES的base64编码
                "encrypted_image": enc_pic,  # 加密后的图片
                "signature": base64.b64encode(signature).decode('utf-8'),  # 签名
                "encrypted_aes_key": enckey_str,  # 加密的AES密钥
                "aes_key": aes_key,  # 原始AES密钥（仅用于测试）
                "mes_size": len(MES)
            }
        })

    except Exception as e:
        print(f"加密过程中发生错误: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"加密失败: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)