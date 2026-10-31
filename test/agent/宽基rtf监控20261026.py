import akshare as ak
import schedule
import time
import csv
import requests
from datetime import datetime

# ====================== 【配置区，修改这里】 ======================
watch_etf = [
    {"code": "510050", "name": "上证50ETF", "is_qdii": False},
    {"code": "510300", "name": "沪深300ETF", "is_qdii": False},
    {"code": "159601", "name": "中证A50ETF", "is_qdii": False},
    {"code": "510500", "name": "中证500ETF", "is_qdii": False},
    {"code": "560510", "name": "中证A500ETF", "is_qdii": False},
    {"code": "159845", "name": "中证1000ETF", "is_qdii": False},
    {"code": "159915", "name": "创业板ETF", "is_qdii": False},
    {"code": "588000", "name": "科创50ETF", "is_qdii": False},
    {"code": "513500", "name": "标普500ETF", "is_qdii": True},
    {"code": "513100", "name": "纳指100ETF", "is_qdii": True},
    {"code": "159920", "name": "恒生ETF", "is_qdii": True},
]
INTERVAL_MINUTE = 5
# 告警阈值（单位：万元）
ALERT_MAIN_IN = 10000        # 全天累计主力净流入>1亿 告警
ALERT_INCREMENT_IN = 3000     # 5分钟增量流入>3000万告警
ALERT_INCREMENT_OUT = -3000   # 【新增】5分钟增量流出<-3000万告警
VOLUME_MULTIPLE = 2.0         # 成交额 >= 昨日全天2倍，放量告警
PRICE_CHANGE_THRESHOLD = 3.0  # 涨跌幅告警阈值 ±3%
LOG_FILE = "etf_log.csv"

# ========== 企业微信机器人，粘贴你的webhook链接 ==========
WECOM_WEBHOOK = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=这里替换成你的key"
SEND_FULL_SNAPSHOT = True # 是否每5分钟推送完整表格到企微
SEND_ALERT = True         # 是否异动单独推送告警消息
# ==========================================================

# ANSI颜色码（终端打印颜色）
RED = "\033[91m"
RESET = "\033[0m"

# 全局变量
last_snapshot = {}          # 保存上一轮主力净流入，用于增量计算
alert_record = []           # 存储当日所有告警记录，收盘汇总使用

def is_trade_time():
    """判断当前是否A股盘中交易时间"""
    now = datetime.now()
    wd = now.weekday()
    # 周六、周日不运行
    if wd >= 5:
        return False
    current_time = now.time()
    morning_start = datetime.strptime("09:30", "%H:%M").time()
    morning_end = datetime.strptime("11:30", "%H:%M").time()
    afternoon_start = datetime.strptime("13:00", "%H:%M").time()
    afternoon_end = datetime.strptime("15:00", "%H:%M").time()
    if (morning_start <= current_time <= morning_end) or (afternoon_start <= current_time <= afternoon_end):
        return True
    return False

def is_close_time():
    """判断是否15:00收盘，触发收盘总结"""
    now = datetime.now()
    wd = now.weekday()
    if wd >=5:
        return False
    ct = now.time()
    close_time = datetime.strptime("15:00", "%H:%M").time()
    return ct.hour == close_time.hour and ct.minute == close_time.minute

def send_wecom_msg(content):
    """企业微信机器人推送消息"""
    headers = {"Content-Type": "application/json"}
    data = {
        "msgtype": "markdown",
        "markdown": {
            "content": content
        }
    }
    try:
        res = requests.post(WECOM_WEBHOOK, json=data, headers=headers, timeout=10)
        return res.json()
    except Exception as e:
        print(f"企微推送失败: {e}")
        return None

def get_etf_last_day_amount(code):
    """获取ETF昨日全天成交额（万元）"""
    try:
        df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date="", end_date="", adjust="")
        last_day_amt = float(df.iloc[-2]["成交额"]) * 10000
        return last_day_amt
    except:
        return None

def fetch_etf_data():
    result_rows = []
    for item in watch_etf:
        code = item["code"]
        name = item["name"]
        is_qdii = item["is_qdii"]
        try:
            df = ak.fund_etf_realtime_em(symbol=code)
            now_price = float(df.loc[0, "净值"])
            change_pct = float(df.loc[0, "涨跌幅"])
            total_amount = float(df.loc[0, "成交额"])
            main_net_in = float(df.loc[0, "主力净流入-净额"])
            big_buy = float(df.loc[0, "超大单净流入-净额"])
            big_sell = float(df.loc[0, "大单净流入-净额"])
            last_day_amt = get_etf_last_day_amount(code)

            # 计算5分钟增量资金
            increment = 0.0
            if code in last_snapshot:
                prev_main = last_snapshot[code]
                increment = main_net_in - prev_main

            result_rows.append({
                "ETF代码": code,
                "ETF名称": name,
                "is_qdii": is_qdii,
                "最新价": now_price,
                "涨跌幅%": change_pct,
                "累计成交额(万)": round(total_amount, 2),
                "超大单净额(万)": round(big_buy, 2),
                "大单净额(万)": round(big_sell, 2),
                "主力净流入(万)": round(main_net_in, 2),
                "5分钟增量资金(万)": round(increment,2),
                "昨日成交额(万)": last_day_amt
            })
            # 更新全局快照
            last_snapshot[code] = main_net_in
        except Exception as e:
            result_rows.append({
                "ETF代码": code,
                "ETF名称": name,
                "is_qdii": is_qdii,
                "最新价": -1,
                "涨跌幅%": -1,
                "累计成交额(万)": -1,
                "超大单净额(万)": -1,
                "大单净额(万)": -1,
                "主力净流入(万)": -1,
                "5分钟增量资金(万)": -1,
                "昨日成交额(万)": None
            })
    return result_rows


def check_alert(row):
    alert_msg = []
    if row["is_qdii"] is True:
        return alert_msg
    main_in = row["主力净流入(万)"]
    increment_in = row["5分钟增量资金(万)"]
    curr_vol = row["累计成交额(万)"]
    last_vol = row["昨日成交额(万)"]
    change_pct = row["涨跌幅%"]

    # 全天累计资金流入告警
    if main_in > ALERT_MAIN_IN:
        alert_msg.append(f"【累计资金异动】主力净流入 {main_in:.0f}万 超过阈值{ALERT_MAIN_IN}万")
    # 5分钟增量流入告警
    if increment_in > ALERT_INCREMENT_IN:
        alert_msg.append(f"【5分钟增量流入】新增流入{increment_in:.0f}万，>{ALERT_INCREMENT_IN}万")
    # 【新增】5分钟增量流出砸盘告警
    if increment_in < ALERT_INCREMENT_OUT:
        alert_msg.append(f"【5分钟增量流出】快速出逃{increment_in:.0f}万，<{ALERT_INCREMENT_OUT}万")
    # 放量告警
    if last_vol and curr_vol > 0 and curr_vol >= last_vol * VOLUME_MULTIPLE:
        alert_msg.append(f"【放量异动】当前成交额 {curr_vol:.0f}万，达到昨日{VOLUME_MULTIPLE}倍")
    # 涨跌幅告警
    if abs(change_pct) >= PRICE_CHANGE_THRESHOLD:
        if change_pct > 0:
            alert_msg.append(f"【大涨】涨跌幅{change_pct:.2f}%，超过+{PRICE_CHANGE_THRESHOLD}%")
        else:
            alert_msg.append(f"【大跌】涨跌幅{change_pct:.2f}%，低于-{PRICE_CHANGE_THRESHOLD}%")
    return alert_msg


def save_to_csv(rows, snapshot_time):
    header = ["快照时间","ETF代码","ETF名称","最新价","涨跌幅%","累计成交额(万)","超大单净额(万)","大单净额(万)","主力净流入(万)","5分钟增量资金(万)"]
    write_header = False
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            pass
    except FileNotFoundError:
        write_header = True
    with open(LOG_FILE, 'a', newline="", encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({
                "快照时间": snapshot_time,
                "ETF代码": r["ETF代码"],
                "ETF名称": r["ETF名称"],
                "最新价": r["最新价"],
                "涨跌幅%": r["涨跌幅%"],
                "累计成交额(万)": r["累计成交额(万)"],
                "超大单净额(万)": r["超大单净额(万)"],
                "大单净额(万)": r["大单净额(万)"],
                "主力净流入(万)": r["主力净流入(万)"],
                "5分钟增量资金(万)": r["5分钟增量资金(万)"],
            })

def build_wechat_table(rows, snapshot_time):
    """构建企微消息文本表格（兼容企微markdown不支持原生表格问题）"""
    text = f"### 📊 ETF资金监控快照【{snapshot_time}】\n"
    text += "```\n"
    text += f"{'代码':<8}{'名称':<10}{'现价':<8}{'涨跌%':<8}{'成交额(万)':<12}{'5分钟增量':<12}{'主力净流入':<12}{'告警'}\n"
    text += "-"*105 + "\n"
    for r in rows:
        alerts = check_alert(r)
        alert_text = "; ".join(alerts) if alerts else "-"
        line = (f"{r['ETF代码']:<8}{r['ETF名称']:<10}{r['最新价']:<8.3f}{r['涨跌幅%']:<8.2f}"
                f"{r['累计成交额(万)']:<12.0f}{r['5分钟增量资金(万)']:<12.0f}{r['主力净流入(万)']:<12.0f}{alert_text}")
        text += line + "\n"
    text += "```\n"
    text += "> 说明：正数=资金流入，负数=流出；5分钟增量为本轮相对上一轮新增资金；QDII日间数据仅供参考，资金为估算值。"
    return text

def send_close_summary():
    """收盘汇总推送"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_alerts = len(alert_record)
    msg = f"# 📈【收盘全天汇总】{now}\n"
    msg += f"> 今日总共触发告警：{total_alerts} 条\n\n"
    if total_alerts > 0:
        for item in alert_record:
            msg += f"- {item}\n"
    else:
        msg += "> 今日盘中无告警触发\n"
    msg += "\n> 监控结束，下一个交易日自动重启采集"
    send_wecom_msg(msg)
    # 清空告警记录，留给下个交易日
    alert_record.clear()
    last_snapshot.clear()

def print_markdown_table(rows, snapshot_time):
    print(f"\n===== ETF资金监控【{snapshot_time}】 每5分钟快照 =====")
    save_to_csv(rows, snapshot_time)
    rows_sort = sorted(rows, key=lambda x: x["主力净流入(万)"], reverse=True)

    # 控制台打印
    header = "| ETF代码 | ETF名称 | 最新价 | 涨跌幅% | 累计成交额(万) | 5分钟增量资金(万) | 主力净流入(万) | 告警 |"
    split_line = "| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |"
    print(header)
    print(split_line)

    alert_all = []
    for r in rows_sort:
        alerts = check_alert(r)
        alert_text = "; ".join(alerts) if alerts else "-"
        if alerts:
            for a in alerts:
                alert_line = f"{r['ETF名称']}：{a}"
                alert_all.append(alert_line)
                alert_record.append(f"[{snapshot_time}] {alert_line}") # 存入当日告警记录
            print(f"{RED}| {r['ETF代码']} | {r['ETF名称']} | {r['最新价']:.3f} | {r['涨跌幅%']:.2f} | {r['累计成交额(万)']:.0f} | {r['5分钟增量资金(万)']:.0f} | {r['主力净流入(万)']:.0f} | {alert_text} |{RESET}")
        else:
            print(f"| {r['ETF代码']} | {r['ETF名称']} | {r['最新价']:.3f} | {r['涨跌幅%']:.2f} | {r['累计成交额(万)']:.0f} | {r['5分钟增量资金(万)']:.0f} | {r['主力净流入(万)']:.0f} | {alert_text} |")

    # 推送完整快照
    if SEND_FULL_SNAPSHOT:
        wechat_msg = build_wechat_table(rows_sort, snapshot_time)
        send_wecom_msg(wechat_msg)

    # 异动单独推送告警
    if SEND_ALERT and len(alert_all) > 0:
        alert_str = "\n".join(alert_all)
        alert_msg = f"# ⚠️ETF资金异动告警【{snapshot_time}】\n> {alert_str}"
        send_wecom_msg(alert_msg)
        print(f"\n{RED}!!! 检测到标的异动，企微告警消息已推送 !!!{RESET}")

    print("\n> 说明：主力净流入=超大单+大单；正数流入负数流出；5分钟增量=本次快照-上一轮快照；QDII日间数据仅作参考；资金为估算值。")
    print("="*140)

def monitor_task():
    # 判断是否收盘，发送收盘总结
    if is_close_time():
        send_close_summary()
        return
    if not is_trade_time():
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] 非交易时段，休眠跳过本次抓取")
        return
    data = fetch_etf_data()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_markdown_table(data, now)


if __name__ == "__main__":
    print(f"【终极ETF监控Agent启动】每{INTERVAL_MINUTE}分钟轮询")
    print(f"告警规则：累计主力>{ALERT_MAIN_IN}万｜5分钟流入>{ALERT_INCREMENT_IN}万｜5分钟流出<{ALERT_INCREMENT_OUT}万｜成交额≥昨日{VOLUME_MULTIPLE}倍｜涨跌幅±{PRICE_CHANGE_THRESHOLD}%")
    print(f"企业微信推送开关：全量快照:{SEND_FULL_SNAPSHOT} 异动告警:{SEND_ALERT}，15:00自动收盘总结推送")
    schedule.every(INTERVAL_MINUTE).minutes.do(monitor_task)
    while True:
        schedule.run_pending()
        time.sleep(1)