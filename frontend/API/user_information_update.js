function update() {
  var settings = {
    "url": "http://localhost:8080/user/update",
    "method": "PUT",
    "timeout": 3000,
    "headers": {
      "Authorization": "用户登录返回的Token",
      "Content-Type": "application/json"
    },
    "data": JSON.stringify({
      "username": "string",
      "email": "string",
      "phoneNumber": "string",
      "avatarUrl": "string"
    }),
  };
  var updateRequest = $.ajax(settings)
  updateRequest.done(function (response) {
    console.log(response);
  });
  updateRequest.fail(function (error) {
    console.log(error)
  })
  return updateRequest
}
export { update }