
export class AIService {
  static async getClinicalAdvice(reportData) {
    const prompt = `请根据以下眼科诊断报告生成临床建议：
诊断时间：${new Date(reportData.time).toLocaleString()}
患者信息：${reportData.username}
主要诊断结果：${Object.entries(reportData.report.predictionResultsList[0].diseases)
        .map(([disease, info]) => `${disease} (${(info.probability * 100).toFixed(2)}%)`)
        .join(', ')}

请用中文给出专业的临床处理建议，包括：
1. 建议的进一步检查
2. 治疗方案建议
3. 日常护理指导
4. 复诊建议`;

    return new Promise((resolve) => {
      console.log('Sending request with payload:', { prompt }); // 添加请求日志
      $.ajax({
        url: `http://8.137.104.3:8082/deepseek/interact?prompt=${encodeURIComponent(prompt)}`,
        method: 'POST',
        // method: 'POST',
        timeout: 1000000,
        headers: {
          'Authorization': localStorage.getItem('token'),
          'Content-Type': 'application/json' // 修复缺失的内容类型
        },
        // // dataType: 'json',
        // data: JSON.stringify({ prompt }),
        success: (response) => resolve(response),
        error: (xhr, status, error) => {
          console.error('AJAX Error Detail:', {
            status: xhr.status,
            statusText: xhr.statusText,
            responseText: xhr.responseText,
            error: error
          });
          resolve({
            error: "服务暂时不可用，请联系管理员",
            debug: `HTTP ${xhr.status}: ${xhr.responseText || '无响应内容'}`
          });
        }
      });
    });
  }

  static async chatWithAI(messages) {
    return new Promise((resolve) => {
      $.ajax({
        url: `http://8.137.104.3:8082/deepseek/interact?prompt=${messages[messages.length - 1].content}`,
        method: 'POST',
        timeout: 1000000,
        headers: {
          'Authorization': localStorage.getItem('token'),
          'Content-Type': 'application/json' // 修复内容类型
        },
        // dataType: 'json',
        // data: JSON.stringify(messages), // 修正请求体格式
        success: (response) => resolve(response),
        error: (xhr) => {
          const errorMsg = xhr.responseJSON?.message || xhr.statusText;
          console.error("AI请求失败:", errorMsg);
          resolve({
            error: "AI服务暂时不可用，请稍后重试",
            details: `错误代码 ${xhr.status}: ${errorMsg}`
          });
        }
      });
    });
  }
}