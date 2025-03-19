function user_details() {
  var settings = {
    "url": "http://localhost:8080/user/details",
    "method": "GET",
    "timeout": 2000,
    "headers": {
      "Authorization": "用户登录返回的Token",
      "Content-Type": "application/json"
    },
  };
  var user_detailsRequest = $.ajax(settings)
  user_detailsRequest.done(function (response) {
    console.log(response);
  });
  user_detailsRequest.fail(function (error) {
    console.log(error)
  })
  return user_detailsRequest
}
export { user_details }