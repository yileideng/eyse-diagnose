function diagnose() {
  var form = new FormData();
  form.append("fileList", fileInput.files[0], "C:\\path\\ppp");
  form.append("fileList", fileInput.files[0], "C:\\path\\ppp");
  form.append("fileList", fileInput.files[0], "C:\\path\\ppp");

  var settings = {
    "url": "http://localhost:8080/diagnose/upload/image",
    "method": "POST",
    "timeout": 5000,
    "headers": {
      "Authorization": "登录后返回的Token"
    },
    "processData": false,
    "mimeType": "multipart/form-data",
    "contentType": false,
    "data": form
  };
  var diagnoseRequest = $.ajax(settings)
  diagnoseRequest.done(function (response) {
    console.log(response);
  });
  diagnoseRequest.fail(function (error) {
    console.log(error)
  })
  return diagnoseRequest
}
export { diagnose }