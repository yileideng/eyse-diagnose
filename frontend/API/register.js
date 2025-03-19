function register() {
  var settings = {
    "url": "http://localhost:8080/register",
    "method": "POST",
    "timeout": 2000,
    "headers": {
      "Content-Type": "application/json"
    },
    "data": JSON.stringify({
      "username": "string",
      "password": "string",
      "email": "string",
      "phoneNumber": "string",
      "avatarUrl": "string"
    }),
  };
  var registerRequest = $.ajax(settings)
  registerRequest.done(function (response) {
    console.log(response);
  });
  registerRequest.fail(function (error) {
    console.log(error)
  })
  return registerRequest
}
export { register }