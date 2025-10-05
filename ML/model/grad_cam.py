import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # 使用正确的钩子注册方法
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))
    
    def generate_cam(self, input_image, target_class):
        # 清理之前挂接的钩子
        self.gradients = None
        self.activations = None
        
        # 确保输入需要梯度
        input_image = input_image.requires_grad_(True)
        
        # 确保模型处于正确的模式
        self.model.eval()
        
        # 获取输出和特征
        try:
            result = self.model(input_image, grad_cam=True, apply_health_constraint=False)
        except TypeError:
            result = self.model(input_image, grad_cam=True)
        if isinstance(result, tuple):
            output = result[0]
        else:
            output = result
        
        # 确保目标类是有效的
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # 创建一个one-hot编码的目标
        one_hot = torch.zeros_like(output)
        one_hot[:, int(target_class)] = 1
        
        # 清除之前的梯度
        self.model.zero_grad()
        
        # 反向传播
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 检查钩子是否正确捕获了梯度
        if self.gradients is None:
            raise ValueError("梯度为None，钩子可能未正确捕获梯度")
        
        # 计算权重并应用权重优化
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # 使用全局平均池化计算权重，并应用ReLU激活
        weights = F.relu(torch.mean(gradients, dim=(2, 3), keepdim=True))
        
        # 应用权重到激活图
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # 应用ReLU
        cam = F.relu(cam)
        
        # 上采样到原始图像大小
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # 应用阈值过滤，只保留高激活值区域
        threshold = 0.5  # 设置阈值
        cam[cam < threshold] = 0
        
        # 重新归一化
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.squeeze().cpu().numpy()
    
    def overlay_cam(self, image, cam, alpha=0.5):
        # 将图像转换为RGB格式（如果需要）
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 将CAM转换为热力图
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # 将热力图叠加到原始图像上
        overlay = cv2.addWeighted(image, 1-alpha, cam, alpha, 0)
        
        return overlay
        
    def __del__(self):
        # 清理钩子
        for hook in self.hooks:
            hook.remove()