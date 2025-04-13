function uploadZip(formData) {
  const settings = {
    "url": "http://8.137.104.3:8082/diagnose/upload/zip",  // 你的zip接口地址
    "method": "POST",
    "timeout": 50000,
    "headers": {
      "Authorization": localStorage.getItem('token')  // 携带token
    },
    "processData": false,
    "mimeType": "multipart/form-data",
    "contentType": false,
    "data": formData  // 直接使用传入的formData
  };

  return $.ajax(settings);
}

export { uploadZip };  // 命名导出函数