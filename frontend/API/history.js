function history() {
  var settings = {
    "url": "http://localhost:8080/diagnose/history",
    "method": "POST",
    "timeout": 3000,
    "headers": {
      "Authorization": "登录后返回的Token",
      "Content-Type": "application/json"
    },
    "data": JSON.stringify({
      "pageNo": "1",
      "pageSize": "5",
      "sortBy": "time",
      "isAsc": "false"
    }),
  };

  var historyRequest = $.ajax(settings)
  historyRequest.done(function (response) {
    console.log(response);
  });
  historyRequest.fail(function (error) {
    console.log(error)
  })
  return historyRequest
}
export { history }