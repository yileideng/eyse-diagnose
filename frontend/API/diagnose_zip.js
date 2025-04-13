
function diagnoseZip(zipIds) {
  var settings = {
    "url": "http://8.137.104.3:8082/diagnose/create-bulk",
    "method": "POST",
    "timeout": 50000,
    "headers": {
      "Authorization": localStorage.getItem('token'),
      "Content-Type": "application/json"
    },
    "data": JSON.stringify(zipIds)
  };
  return $.ajax(settings)
}
export { diagnoseZip }