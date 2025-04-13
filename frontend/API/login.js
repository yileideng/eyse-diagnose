function login(username, password) {
  return $.ajax({
    url: "http://8.137.104.3:8082/login",
    method: "POST",
    timeout: 2000,
    headers: { "Content-Type": "application/json" },
    data: JSON.stringify({ username, password }) // 动态传递参数
  });
}
//把对象保存在变量中保证只请求一次：
//   var loginRequest = $.ajax(settings);
//   loginRequest.done(function (response) {
//     console.log(response);
//     //成功的情况

//   });
//   loginRequest.fail(function (error) {
//     console.error("Login failed!", error);

//   })
//   return loginRequest;
// }
export { login };