function opinion(imageIds) {  // 接收动态图片ID参数
  const settings = {
    "url": "http://8.137.104.3:8082/diagnose/create-personal",
    "method": "POST",
    "timeout": 2000000,
    "headers": {
      "Authorization": localStorage.getItem('token'), // 从本地存储获取真实token
      "Content-Type": "application/json"
    },
    "data": JSON.stringify(imageIds) // 使用传入的ID数组
  };

  return $.ajax(settings); // 直接返回Promise对象
}

export { opinion };