let currentRow = 0; // 当前展示的行索引  
const rows = document.querySelectorAll('.row'); // 获取所有的行  
const totalRows = rows.length; // 总的行数量  
const rowsToShow = 2; // 每次展示的行数量  

// 初始化显示的行  
function initDisplay() {
  for (let i = 0; i < Math.min(rowsToShow, totalRows); i++) {
    rows[i].classList.add('show'); // 添加显示类  
  }
}

// 显示新的行  
function showRows(direction) {
  // 隐藏当前的行  
  for (let i = 0; i < rowsToShow; i++) {
    if (currentRow + i < totalRows) {
      rows[currentRow + i].classList.remove('show'); // 移除显示类  
      rows[currentRow + i].classList.add('hide'); // 添加隐藏类  
    }
  }

  // 更新当前行索引  
  if (direction === 'down' && currentRow + rowsToShow < totalRows) {
    currentRow += rowsToShow; // 向下滚动，增加索引  
  } else if (direction === 'up' && currentRow > 0) {
    currentRow -= rowsToShow; // 向上滚动，减少索引  
  }

  // 显示新的行  
  for (let i = 0; i < rowsToShow; i++) {
    if (currentRow + i < totalRows) {
      rows[currentRow + i].classList.remove('hide'); // 移除隐藏类  
      rows[currentRow + i].classList.add('show'); // 添加显示类  
    }
  }

  // 确保当前显示的内容在 container 顶部  
  const container = document.querySelector('.container');
  if (currentRow < totalRows) {
    const currentContents = container.children[currentRow];
    if (currentContents) {
      currentContents.scrollIntoView({ behavior: 'smooth', block: 'start' }); // 滚动到当前行的顶部  
    }
  }
}

// 防抖函数  
function debounce(func, delay) {
  let timeout;
  return function (...args) {
    const context = this;
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(context, args), delay);
  };
}

// 滚动事件  
window.addEventListener('wheel', debounce(function (event) {
  if (event.deltaY > 0) { // 向下滚动  
    showRows('down');
  } else { // 向上滚动  
    showRows('up');
  }
}, 300)); // 设置防抖时间为300毫秒  

// 初始化  
initDisplay();  