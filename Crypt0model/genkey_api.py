from flask import Flask, jsonify
from algorithm.RSAalgorithm import generateKeys

app = Flask(__name__)

@app.route('/generate-user-keys', methods=['GET'])
def generate_user_keys():
    private_key, public_key = generateKeys()

    return jsonify({
        "private_key": private_key,
        "public_key": public_key
    })


@app.route('/generate-model-keys', methods=['GET'])
def generate_model_keys():
    private_key, public_key = generateKeys()

    return jsonify({
        "private_key": private_key,
        "public_key": public_key
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)