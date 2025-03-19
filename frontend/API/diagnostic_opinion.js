function opinion() {
  var settings = {
    "url": "http://localhost:8080/diagnose/create",
    "method": "POST",
    "timeout": 2000,
    "headers": {
      "Authorization": "登录后返回的Token",
      "Content-Type": "application/json"
    },
    "data": JSON.stringify({
      "urlList": [
        "string"
      ]
    }),
  };
  var opinionRequest = $.ajax(settings)
  opinionRequest.done(function (response) {
    console.log(response);
  });
  opinionRequest.fail(function (error) {
    console.log(error)
  })
  return opinionRequest
}
export { opinion }