function logout() {
  var settings = {
    "url": "http://localhost:8080/user/logout",
    "method": "GET",
    "timeout": 1000,
    "headers": {
      "Authorization": "登录后返回的token",
      "Content-Type": "application/json"
    },
  };
  var logoutrequest = $.ajax(settings)
  logoutrequest.done(function (response) {
    console.log(response);

  });
  logoutrequest.fail(function (error) {
    console.log(error)
  })
  return logoutrequest
}
export { logout }