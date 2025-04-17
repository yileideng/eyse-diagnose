function update(data) {
  const settings = {
    "url": "http://8.137.104.3:8082/user/update",
    "method": "PUT",
    "timeout": 3000,
    "headers": {
      "Authorization": localStorage.getItem('token'),
      "Content-Type": "application/json"
    },
    "data": JSON.stringify(data)
  };

  return $.ajax(settings);
}
export { update }