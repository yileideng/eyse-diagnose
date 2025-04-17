function sendmail(email) {

  const settings = {
    "url": `http://8.137.104.3:8082/mail?mail=${email}`,
    "method": "GET",
    "timeout": 5000,
    "Access-Control-Allow-Origin": "*",
    "headers": {
      // "Authorization": localStorage.getItem('token'), // 使用存储的token
      "Content-Type": "application/json"
    },
    // "data": email
  };

  return $.ajax(settings);
}
export { sendmail }