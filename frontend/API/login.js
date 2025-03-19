function login(username, password) {
  var settings = {
    "url": "http://localhost:8080/login",
    "method": "POST",
    "timeout": 1000,
    "headers": {
      "Content-Type": "application/json"
    },
    "data": JSON.stringify({
      "username": username,
      "password": password
    }),
  };
  //把对象保存在变量中保证只请求一次：
  var loginRequest = $.ajax(settings);
  loginRequest.done(function (response) {
    console.log(response);
    //成功的情况

  });
  loginRequest.fail(function (error) {
    console.error("Login failed!", error);

  })
  return loginRequest;
}
export { login };