// register.js
function register(username, password, email, code) {
  return $.ajax({
    url: "http://8.137.104.3:8082/register",
    method: "POST",
    timeout: 2000,
    headers: { "Content-Type": "application/json" },
    data: JSON.stringify({
      username, password, email, code,

    }) // 动态传递参数
  });
}
//   var registerRequest = $.ajax(settings)
//   registerRequest.done(function (response) {
//     console.log(response);
//   });
//   registerRequest.fail(function (error) {
//     console.log(error)
//   })
//   return registerRequest
// }
export { register }