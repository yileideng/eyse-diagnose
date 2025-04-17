function diagnose_details(id) {
  var settings = {
    "url": `http://8.137.104.3:8082/diagnose/details?diagnoseId=${id}`,
    "method": "GET",
    "timeout": 3000,
    "headers": {
      "Authorization": localStorage.getItem('token'),
      "Content-Type": "application/json"
    },
  };

  return $.ajax(settings)
}
export { diagnose_details }