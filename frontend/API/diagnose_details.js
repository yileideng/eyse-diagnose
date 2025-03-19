function diagnose_details() {
  var settings = {
    "url": "http://localhost:8080/diagnose/details?diagnoseId=1",
    "method": "GET",
    "timeout": 3000,
    "headers": {
      "Authorization": "用户登录后返回的Token",
      "Content-Type": "application/json"
    },
  };

  var diagnose_detailsRequest = $.ajax(settings)
  diagnose_detailsRequest.done(function (response) {
    console.log(response);
  });
  diagnose_detailsRequest.fail(function (error) {
    console.log(error)
  })
  return diagnose_detailsRequest
}
export { diagnose_details }