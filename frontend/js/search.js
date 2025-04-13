document.addEventListener('DOMContentLoaded', () => {
  // 获取搜索相关元素
  const searchInput = document.querySelector('.search-input');
  const searchIcon = document.querySelector('.search-icon');

  // 创建存储报告索引的Map
  let reportMap = new Map();

  // 初始化报告映射
  function initReportMap() {
    document.querySelectorAll('.right').forEach(right => {
      const index = right.getAttribute('data-index');
      const resultIds = JSON.parse(localStorage.getItem('resultIds')) || [];
      if (resultIds[index - 1]) {
        reportMap.set(resultIds[index - 1].resultId, right);
      }
    });
  }

  // 搜索功能实现
  async function handleSearch(reportId) {
    if (!reportId) {
      alert('请输入有效的报告编号');
      return;
    }

    // 显示加载状态
    searchIcon.textContent = '⏳';

    try {
      // 检查本地缓存
      const targetRight = reportMap.get(reportId);

      if (targetRight) {
        // 直接触发对应detail点击
        const detail = targetRight.querySelector('.detail');
        detail.click();
      } else {
        // 尝试从服务器获取
        const response = await diagnose_details(reportId);
        if (response.code === 200) {
          // 创建临时卡片显示结果
          createTempCard(response.data);
        } else {
          alert('未找到该报告编号');
        }
      }
    } catch (error) {
      console.error('搜索失败:', error);
      alert(error.status ? `服务器错误: ${error.status}` : '未能找到该编号，请检查输入是否正确或网络是否正常');
    } finally {
      searchIcon.textContent = '🔍';
    }
  }

  // 创建临时结果卡片
  function createTempCard(data) {
    const tempRight = document.createElement('div');
    tempRight.className = 'right';
    tempRight.innerHTML = `
      <div class="summary-view">
          <div class="diagnose-time">${new Date(data.time).toLocaleString()}</div>
          <div class="main-disease">
              <span class="disease-name">${getMainDisease(data)}</span>
              <span class="probability">${getMaxProbability(data)}%</span>
          </div>
      </div>
      <div class="detail"></div>
  `;

    // 绑定点击事件
    tempRight.querySelector('.detail').addEventListener('click', () => {
      showSearchResult(data, tempRight);
    });

    // 创建临时行容器
    const tempRow = document.createElement('div');
    tempRow.className = 'row show';
    tempRow.appendChild(document.createElement('div')); // 空的left元素
    tempRow.appendChild(tempRight);

    // 插入到容器顶部
    document.querySelector('.container').prepend(tempRow);
    tempRight.querySelector('.detail').click();
  }

  // 显示搜索结果
  async function showSearchResult(data, targetRight) {
    const row = targetRight.parentElement;
    targetRight.innerHTML = '<div class="loading-spinner"></div>';

    try {
      const detailsContent = `
          <div class="diagnose-content">
              ${generateSummaryView(data)}
              ${generateFullTable(data)}
          </div>
          <div class="detail"></div>
      `;

      targetRight.innerHTML = detailsContent;
      openExpandedView(targetRight, row);
    } catch (error) {
      targetRight.innerHTML = `
          <div class="error">报告加载失败<br>${error.message}</div>
          <div class="detail"></div>
      `;
    }
  }

  // 辅助函数
  function getMainDisease(data) {
    const diseases = data.report.predictionResultsList[0].diseases;
    return Object.entries(diseases).reduce((a, b) =>
      a[1].probability > b[1].probability ? a : b)[0];
  }

  function getMaxProbability(data) {
    const diseases = data.report.predictionResultsList[0].diseases;
    return (Math.max(...Object.values(diseases).map(d => d.probability))) * 100;
  }

  // 事件监听
  searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSearch(searchInput.value.trim());
  });

  // searchIcon.parentElement.addEventListener('click', (e) => {
  //   e.stopPropagation();
  //   handleSearch(searchInput.value.trim());
  // });


  // 初始化报告映射
  initReportMap();
});