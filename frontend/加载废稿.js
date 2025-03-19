$(document).ready(function () {
  // 使用事件委托确保动态加载的内容也能绑定事件
  $(document).on('click', '.nav-link', function (e) {
    e.preventDefault(); // 必须阻止默认跳转行为
    const link = $(this).attr("href");

    // 添加加载状态提示
    $("#content").addClass('transitioning');

    // 使用Promise确保顺序执行
    new Promise((resolve) => {
      $("#content").fadeOut(400, () => resolve());
    }).then(() => {
      // 仅加载目标页面的指定内容区域
      return $("#content").load(link + ' #content > *', function (response, status) {
        if (status === "error") {
          console.log("加载失败");
        }
      });
    }).then(() => {
      // 重新初始化组件
      initComponents();
      $("#content").removeClass('transitioning').fadeIn(400);
    });
  });

  // 初始化页面组件
  function initComponents() {
    // 重新绑定导航事件等初始化操作
    $('[data-toggle="tooltip"]').tooltip();
  }
});




$(document).ready(function () {
  // 使用事件委托确保动态加载内容的事件绑定
  $(document).on('click', '.nav-link', function (e) {
    e.preventDefault(); // 必须阻止默认跳转
    const link = $(this).attr("href");

    // 添加加载状态
    showLoadingIndicator();

    // 分阶段动画流程
    $("#content")
      .fadeOut(300) // 第一阶段：淡出
      .promise().done(function () {
        // 第二阶段：加载内容
        $("#content").load(link + ' #content', function (response, status) {
          if (status === "error") {
            handleLoadingError();
            return;
          }

          // 第三阶段：初始化组件
          initComponents();

          // 第四阶段：淡入动画
          $("#content")
            .css({
              opacity: 0,
              transform: "translateY(20px)"
            })
            .fadeIn(300)
            .animate({
              opacity: 1,
              transform: "translateY(0)"
            }, 600, 'easeOutQuad');

          // 更新浏览器历史记录
          history.pushState({}, '', link);
        });
      });
  });

  // 初始化组件函数
  function initComponents() {
    // 重新绑定导航事件
    $('.nav-link').tooltip('dispose').tooltip();

    // 初始化其他插件
    $('[data-toggle="tooltip"]').tooltip();

    // 重新绑定点击事件（防止重复绑定）
    $('.nav-link').off('click').on('click', function (e) {
      // 原有点击逻辑...
    });
  }

  // 显示加载指示器
  function showLoadingIndicator() {
    const loader = `
  <div class="loading-overlay">
    <div class="spinner"></div>
  </div>`;
    $('body').append(loader);
  }

  // 错误处理
  function handleLoadingError() {
    $("#content").html(`
  <div class="error-message">
    内容加载失败，请<a href="#" onclick="location.reload()">刷新重试</a>
  </div>`).fadeIn();
    $('.loading-overlay').remove();
  }
});