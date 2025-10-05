import requests
import json
import time

def test_health_check(base_url):
    """测试健康检查接口"""
    try:
        response = requests.get(f"{base_url}/health")
        print(f"健康检查: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

def test_prediction_api(base_url, image_path):
    """测试预测API接口"""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{base_url}/api/predict", files=files)
        
        print(f"预测API: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"预测成功: {result['success']}")
            if result['success']:
                print("预测概率:")
                for i, (name, prob) in enumerate(zip(result['class_names'], result['probabilities'])):
                    print(f"  {name}: {prob*100:.2f}%")
                
                if result['detected_diseases']:
                    print(f"检测到的疾病: {result['detected_diseases']}")
                else:
                    print("未检测到明显疾病")
        else:
            print(f"预测失败: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"预测API测试失败: {e}")
        return False

def test_web_interface(base_url):
    """测试Web界面"""
    try:
        response = requests.get(base_url)
        print(f"Web界面: {response.status_code}")
        if response.status_code == 200:
            print("Web界面访问成功")
            return True
        else:
            print("Web界面访问失败")
            return False
    except Exception as e:
        print(f"Web界面测试失败: {e}")
        return False

def main():
    # 配置
    base_url = "http://localhost:5000"  # 根据实际部署地址修改
    test_image_path = "test_image.jpg"  # 测试图片路径
    
    print("开始测试眼科疾病诊断系统API...")
    print(f"测试地址: {base_url}")
    print("-" * 50)
    
    # 测试健康检查
    print("1. 测试健康检查接口")
    health_ok = test_health_check(base_url)
    print()
    
    # 测试Web界面
    print("2. 测试Web界面")
    web_ok = test_web_interface(base_url)
    print()
    
    # 测试预测API
    print("3. 测试预测API接口")
    if health_ok:
        prediction_ok = test_prediction_api(base_url, test_image_path)
    else:
        print("跳过预测测试（健康检查失败）")
        prediction_ok = False
    print()
    
    # 测试结果汇总
    print("=" * 50)
    print("测试结果汇总:")
    print(f"健康检查: {'✓' if health_ok else '✗'}")
    print(f"Web界面: {'✓' if web_ok else '✗'}")
    print(f"预测API: {'✓' if prediction_ok else '✗'}")
    
    if all([health_ok, web_ok, prediction_ok]):
        print("\n🎉 所有测试通过！系统运行正常。")
    else:
        print("\n❌ 部分测试失败，请检查系统配置。")

if __name__ == "__main__":
    main()
