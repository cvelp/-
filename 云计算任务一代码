import os
import requests
import json
import re
import time
from typing import List, Dict, Tuple
from tqdm import tqdm

# 配置参数
MODEL_NAME = "deepseek-r1:7b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"  # Ollama API默认地址
TIMEOUT = 60
RETRIES = 3
NEWS_FILE = "news_data.txt"  # 新闻数据文件
LABELS_FILE = "labels.txt"  # 标签文件
OUTPUT_FILE = "news_analysis_results.json"
LOG_FILE = "model_inference.log"


def log(message: str, level: str = "INFO"):
    """带时间戳的日志输出，同时写入日志文件"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {message}"
    print(log_msg)

    # 写入日志文件
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")


def call_ollama_api(prompt: str) -> str:
    """使用HTTP API调用Ollama模型"""
    start_time = time.time()
    log(f"开始调用模型: {MODEL_NAME}", "DEBUG")

    for attempt in range(RETRIES):
        try:
            # 构建API请求
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False  # 非流式响应
            }

            log(f"发送API请求 (尝试 {attempt + 1}/{RETRIES})", "DEBUG")
            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=TIMEOUT
            )

            duration = time.time() - start_time

            if response.status_code != 200:
                log(f"API调用失败 (状态码 {response.status_code}): {response.text}", "ERROR")
                time.sleep(2)
                continue

            # 解析响应
            data = response.json()
            message_content = data.get("message", {}).get("content", "")

            log(f"模型返回成功 (长度: {len(message_content)} 字符, 耗时: {duration:.2f}秒)", "INFO")
            return message_content

        except requests.Timeout:
            log(f"API请求超时 ({TIMEOUT}秒)", "ERROR")
            continue
        except Exception as e:
            log(f"API调用异常: {type(e).__name__} - {str(e)}", "ERROR")
            time.sleep(2)

    log("达到最大重试次数，返回空结果", "ERROR")
    return ""


def read_news_file() -> List[str]:
    """从文件读取新闻数据"""
    log(f"从文件读取新闻数据: {NEWS_FILE}", "INFO")

    try:
        # 读取新闻数据
        with open(NEWS_FILE, "r", encoding="utf-8") as f:
            news_list = [line.strip() for line in f.readlines() if line.strip()]

        if not news_list:
            log("新闻文件为空，程序退出", "ERROR")
            exit(1)

        log(f"成功读取 {len(news_list)} 条新闻", "INFO")
        return news_list

    except Exception as e:
        log(f"读取新闻文件时出错: {str(e)}", "ERROR")
        exit(1)


def read_labels_file() -> List[int]:
    """从文件读取标签数据"""
    log(f"从文件读取标签数据: {LABELS_FILE}", "INFO")
    try:
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            labels = [int(line.strip()) for line in f.readlines() if line.strip().isdigit()]
        if not labels:
            log("标签文件为空或无有效标签，程序退出", "ERROR")
            exit(1)
        if len(labels) != len(read_news_file()):
            log("新闻数量与标签数量不匹配，程序退出", "ERROR")
            exit(1)
        log(f"成功读取 {len(labels)} 个标签", "INFO")
        return labels
    except Exception as e:
        log(f"读取标签文件时出错: {str(e)}", "ERROR")
        exit(1)


def analyze_news(news_list: List[str], task_type: str) -> List:
    """分析新闻的通用函数，支持不同任务类型"""
    results = []
    log(f"开始执行任务: {task_type}")

    for i, news in enumerate(tqdm(news_list, desc=task_type)):
        # 根据任务类型构建不同的prompt
        if task_type == "真假判别":
            prompt = f"""
            请判断以下新闻的真假，仅输出0或1：
            新闻：{news}
            """
        elif task_type == "情感分析":
            prompt = f"""
            请分析以下新闻的情感倾向，仅输出"积极"、"消极"或"中性"：
            新闻：{news}
            """
        else:
            log(f"不支持的任务类型: {task_type}", "ERROR")
            continue

        # 添加任务标识符到日志
        log(f"处理新闻 {i + 1}/{len(news_list)} (任务: {task_type})", "INFO")

        response = call_ollama_api(prompt)

        if task_type == "真假判别":
            result = parse_prediction(response)
        elif task_type == "情感分析":
            result = parse_sentiment(response)

        results.append(result)

        # 每处理5条新闻，显示一次进度
        if (i + 1) % 5 == 0:
            log(f"已处理 {i + 1}/{len(news_list)} 条新闻", "INFO")

    return results


def parse_prediction(response: str) -> int:
    """解析模型返回的真假判别结果"""
    match = re.search(r'\b(0|1)\b', response)
    if match:
        log(f"解析预测结果: {match.group(1)}", "DEBUG")
        return int(match.group(1))
    else:
        log(f"无法解析预测结果: {response[:50]}...", "WARNING")
        return 0


def parse_sentiment(response: str) -> str:
    """解析模型返回的情感分析结果"""
    if re.search(r'\b(积极|正面|好|支持)\b', response, re.IGNORECASE):
        return "积极"
    elif re.search(r'\b(消极|负面|坏|反对)\b', response, re.IGNORECASE):
        return "消极"
    else:
        return "中性"


def calculate_accuracy(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """计算准确率指标"""
    total = len(labels)
    if total == 0:
        return {"Accuracy": 0, "Accuracy_fake": 0, "Accuracy_true": 0}

    correct = sum(1 for p, t in zip(predictions, labels) if p == t)
    total_fake = sum(1 for t in labels if t == 0)
    total_true = sum(1 for t in labels if t == 1)

    correct_fake = sum(1 for p, t in zip(predictions, labels) if p == 0 and t == 0)
    correct_true = sum(1 for p, t in zip(predictions, labels) if p == 1 and t == 1)

    accuracy = correct / total
    accuracy_fake = correct_fake / total_fake if total_fake > 0 else 0
    accuracy_true = correct_true / total_true if total_true > 0 else 0

    return {
        "Accuracy": accuracy,
        "Accuracy_fake": accuracy_fake,
        "Accuracy_true": accuracy_true
    }


def main():
    print("\n=== 离线DeepSeek新闻分析系统 ===")
    print(f"使用模型: {MODEL_NAME}")
    print(f"API地址: {OLLAMA_API_URL}")
    print(f"新闻文件: {NEWS_FILE}")
    print(f"标签文件: {LABELS_FILE}")
    print("=" * 50)

    # 检查Ollama API是否可用
    try:
        response = requests.get(f"{OLLAMA_API_URL}/../tags", timeout=10)
        if response.status_code != 200:
            log(f"无法连接到Ollama API: {response.text}", "ERROR")
            exit(1)

        models = [model["name"] for model in response.json().get("models", [])]
        if MODEL_NAME not in models:
            log(f"未找到模型 {MODEL_NAME}，请确保已正确下载", "ERROR")
            exit(1)

        log(f"成功连接到Ollama API，可用模型: {', '.join(models)}", "INFO")
    except Exception as e:
        log(f"检查Ollama API时出错: {str(e)}", "ERROR")
        exit(1)

    # 从文件读取新闻
    news_list = read_news_file()
    # 从文件读取标签
    labels = read_labels_file()

    # 任务1：基础真假判别
    print("\n=== 执行任务1：基础真假判别 ===")
    predictions = analyze_news(news_list, "真假判别")

    # 计算准确率
    accuracy_metrics = calculate_accuracy(predictions, labels)
    print("\n=== 准确率计算结果 ===")
    print(f"Accuracy: {accuracy_metrics['Accuracy']}")
    print(f"Accuracy_fake: {accuracy_metrics['Accuracy_fake']}")
    print(f"Accuracy_true: {accuracy_metrics['Accuracy_true']}")

    # 任务2：情感分析
    print("\n=== 执行任务2：情感分析 ===")
    sentiments = analyze_news(news_list, "情感分析")

    # 保存结果
    results = {
        "news_data": news_list,
        "predictions": predictions,
        "sentiments": sentiments,
        "accuracy_metrics": accuracy_metrics
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {OUTPUT_FILE}")
    print(f"详细日志已保存到: {LOG_FILE}")
    print("分析完成！")


if __name__ == "__main__":
    main()
