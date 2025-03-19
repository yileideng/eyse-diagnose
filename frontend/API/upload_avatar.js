function uploadAvatar() {
  var form = new FormData();
  form.append("file", fileInput.files[0], "");

  var settings = {
    "url": "http://localhost:8080/upload/avatar",
    "method": "POST",
    "timeout": 1000,
    "processData": false,
    "mimeType": "multipart/form-data",
    "contentType": false,
    "data": form
  };
  var avatarRequest = $.ajax(settings)
  avatarRequest.done(function (response) {
    console.log(response);
  });

  avatarRequest.fail(function (error) {
    console.log(error)
  })
  return avatarRequest
}
export { uploadAvatar }