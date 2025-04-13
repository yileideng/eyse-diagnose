function diagnose(formData) { // 接收formData参数
  var settings = {
    "url": "http://8.137.104.3:8082/diagnose/upload/image",
    "method": "POST",
    "timeout": 50000,
    "headers": {
      "Authorization": localStorage.getItem('token')// 从本地存储获取token
    },
    "processData": false,
    "mimeType": "multipart/form-data",
    "contentType": false,
    "data": formData
  };

  return $.ajax(settings);
}

export { diagnose }