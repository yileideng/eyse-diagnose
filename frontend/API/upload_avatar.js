// import { json } from "express";

let cropper = null;
let currentFile = null;

function avatarUpload(form) {
  var settings = {
    "url": "http://8.137.104.3:8082/upload/avatar",
    "method": "POST",
    "timeout": 10000,
    "headers": {
      "Authorization": localStorage.getItem('token')
    },
    "processData": false,
    "mimeType": "multipart/form-data",
    "contentType": false,
    "data": form
  };

  return $.ajax(settings);
}

function initAvatarUpload() {
  const profile = document.querySelector('.profile');
  const modal = document.querySelector('.avatar-modal');
  const closeBtn = modal.querySelector('.close');
  const saveBtn = modal.querySelector('.confirm');
  const input = document.querySelector('#avatar-input');
  // 点击头像打开模态框
  profile.addEventListener('click', () => {
    modal.style.display = 'flex';
  });

  // 关闭模态框
  const closeModal = () => {
    modal.style.display = 'none';
    if (cropper) {
      cropper.destroy();
      cropper = null;
    }
  };

  closeBtn.addEventListener('click', closeModal);

  // 文件选择处理
  input.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    currentFile = file;
    const reader = new FileReader();
    reader.onload = (event) => {
      const image = document.getElementById('cropper-image');
      image.src = event.target.result;

      if (cropper) cropper.destroy();

      cropper = new Cropper(image, {
        aspectRatio: 1,
        viewMode: 1,
        guides: false,
        background: false,
        autoCropArea: 0.8,
        cropBoxResizable: false,
        preview: '.profile svg', // 使用profile区域作为预览
        crop: () => {
          // 圆形裁剪效果
          const canvas = cropper.getCroppedCanvas({
            imageSmoothingQuality: 'high',
            fillColor: 'transparent'
          });

          const context = canvas.getContext('2d');
          context.beginPath();
          context.arc(canvas.width / 2, canvas.height / 2, canvas.width / 2, 0, Math.PI * 2);
          context.closePath();
          context.clip();
        }
      });
    };
    reader.readAsDataURL(file);
  });

  // 保存处理
  saveBtn.addEventListener('click', async () => {
    if (!cropper) return;

    const canvas = cropper.getCroppedCanvas({
      width: 200,
      height: 200
    });

    canvas.toBlob(async (blob) => {
      const file = new File([blob], `avatar_${Date.now()}.jpg`, {
        type: 'image/jpeg',
      });

      // 更新本地预览
      const reader = new FileReader();
      reader.onload = (e) => {
        const svg = document.querySelector('.profile svg');
        svg.style.display = 'none';
        document.querySelector('.profile').style.background = `url(${e.target.result}) center/cover`;
      };
      reader.readAsDataURL(blob);

      // 上传服务器
      const form = new FormData();
      form.append('file', file);

      try {
        const response = await avatarUpload(form);
        const result = JSON.parse(response);
        if (result.code !== 200) throw new Error('上传失败' + result.msg);
        // 保存到LocalStorage
        localStorage.setItem('userAvatar', result.data);

        // 更新头像显示
        updateAvatarDisplay(result.data);
        closeModal();
      } catch (error) {
        console.error('上传错误:', error);
        alert('头像上传失败: ' + error.message);
      }
    }, 'image/jpeg', 0.9);
  });
  // 拖拽上传头像
  const dropZone = modal.querySelector('.modal-body');

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      input.files = files;
      input.dispatchEvent(new Event('change'));
    }
  });
}

function updateAvatarDisplay(url) {
  const profile = document.querySelector('.profile');
  const svg = profile.querySelector('svg');

  // 添加时间戳防止缓存
  const timestamp = new Date().getTime();
  const finalUrl = `${url}?t=${timestamp}`;

  // 创建新的Image对象预加载
  const img = new Image();
  img.src = finalUrl;
  img.onload = () => {
    svg.style.display = 'none';
    profile.style.background = `url(${finalUrl}) center/cover`;
  };
  img.onerror = () => {
    console.error('头像加载失败');
    localStorage.removeItem('userAvatar');
  };
}



// 页面加载时检查存储
document.addEventListener('DOMContentLoaded', () => {
  const savedAvatar = localStorage.getItem('userAvatar');
  if (savedAvatar) {
    updateAvatarDisplay(savedAvatar);
  }
  initAvatarUpload();
});