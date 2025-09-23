import math
import numpy as np
import cv2
from PIL import Image

def _shades_of_gray(image, power=6, gamma=0.8):
    img = image.astype(np.float32)
    img = np.power(img, gamma)
    norm = np.power(img, power)
    norm = np.power(np.mean(norm, axis=(0, 1)), 1 / power)
    scale = np.sqrt(np.sum(norm ** 2))
    norm = norm / (scale + 1e-6)
    img = img / (norm + 1e-6)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def _homomorphic(img):
    f = img.astype(np.float32) / 255.0
    f = np.log1p(f)
    blur = cv2.GaussianBlur(f, (0, 0), sigmaX=10, sigmaY=10)
    g = f - blur
    g = np.expm1(g)
    g = np.clip(g, 0, 1)
    return (g * 255).astype(np.uint8)

def _unsharp(img, strength=1.5, radius=5):
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    out = cv2.addWeighted(img, 1 + strength, blur, -strength, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

def _retina_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    cx = x + w // 2
    cy = y + h // 2
    r = int(max(w, h) * 0.55)
    x1 = max(cx - r, 0)
    y1 = max(cy - r, 0)
    x2 = min(cx + r, image.shape[1])
    y2 = min(cy + r, image.shape[0])
    cropped = image[y1:y2, x1:x2]
    mask = np.zeros_like(gray[y1:y2, x1:x2])
    cv2.circle(mask, (min(r, cropped.shape[1] // 2), min(r, cropped.shape[0] // 2)), int(r * 0.95), 255, -1)
    cropped = cv2.bitwise_and(cropped, cropped, mask=mask)
    return cropped

def preprocess_fundus(pil_image, size=512):
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    image = _retina_crop(image)
    if image.size == 0:
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    image = _shades_of_gray(image)
    image = _homomorphic(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    image = _unsharp(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)
