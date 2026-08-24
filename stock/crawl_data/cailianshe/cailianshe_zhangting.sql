
/**
  我来帮你编写一个完整的财联社首板涨停股票数据抓取和存储系统。这个系统会抓取首板涨停股票，并持续跟踪半年内每天的行情数据
 */

-- 创建数据库
CREATE DATABASE IF NOT EXISTS stock_data DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE stock_data;

-- 首板涨停股票表
CREATE TABLE IF NOT EXISTS first_limit_up_stocks (
    id INT PRIMARY KEY AUTO_INCREMENT,
    stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
    stock_name VARCHAR(50) NOT NULL COMMENT '股票名称',
    first_limit_date DATE NOT NULL COMMENT '首板日期',
    industry VARCHAR(100) COMMENT '所属行业',
    limit_reason TEXT COMMENT '涨停原因',
    concept TEXT COMMENT '所属概念',
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY idx_stock_date (stock_code, first_limit_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='首板涨停股票';

-- 股票每日行情表
CREATE TABLE IF NOT EXISTS stock_daily_data (
    id INT PRIMARY KEY AUTO_INCREMENT,
    stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
    trade_date DATE NOT NULL COMMENT '交易日',
    open_price DECIMAL(10,2) COMMENT '开盘价',
    close_price DECIMAL(10,2) COMMENT '收盘价',
    high_price DECIMAL(10,2) COMMENT '最高价',
    low_price DECIMAL(10,2) COMMENT '最低价',
    volume BIGINT COMMENT '成交量(手)',
    amount DECIMAL(15,2) COMMENT '成交额(万元)',
    change_percent DECIMAL(6,2) COMMENT '涨跌幅(%)',
    limit_up_days INT DEFAULT 0 COMMENT '连板天数',
    is_limit_up TINYINT DEFAULT 0 COMMENT '是否涨停 1:是 0:否',
    limit_reason TEXT COMMENT '当日涨停原因',
    market_cap DECIMAL(15,2) COMMENT '市值(亿元)',
    turnover_rate DECIMAL(6,2) COMMENT '换手率(%)',
    pe_ratio DECIMAL(10,2) COMMENT '市盈率',
    pb_ratio DECIMAL(10,2) COMMENT '市净率',
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY idx_stock_trade (stock_code, trade_date),
    KEY idx_trade_date (trade_date),
    KEY idx_limit_up_days (limit_up_days),
    FOREIGN KEY (stock_code) REFERENCES first_limit_up_stocks(stock_code) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票每日行情';

-- 涨停原因分类表
CREATE TABLE IF NOT EXISTS limit_reason_categories (
    id INT PRIMARY KEY AUTO_INCREMENT,
    reason_type VARCHAR(50) NOT NULL COMMENT '原因类型',
    reason_detail TEXT COMMENT '详细描述',
    stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
    limit_date DATE NOT NULL COMMENT '涨停日期',
    source VARCHAR(100) COMMENT '信息来源',
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    KEY idx_reason_type (reason_type),
    KEY idx_stock_limit (stock_code, limit_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='涨停原因分类';

-- 监控任务表
CREATE TABLE IF NOT EXISTS monitor_tasks (
    id INT PRIMARY KEY AUTO_INCREMENT,
    stock_code VARCHAR(10) NOT NULL COMMENT '股票代码',
    start_date DATE NOT NULL COMMENT '开始监控日期',
    end_date DATE NOT NULL COMMENT '结束监控日期',
    status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
    progress INT DEFAULT 0 COMMENT '进度百分比',
    last_run_date DATE COMMENT '最后运行日期',
    error_message TEXT COMMENT '错误信息',
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_status (status),
    KEY idx_stock_code (stock_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='监控任务';

-- 创建视图：连板统计
CREATE VIEW IF NOT EXISTS continuous_limit_view AS
SELECT
    stock_code,
    stock_name,
    first_limit_date,
    MAX(limit_up_days) as max_continuous_days,
    COUNT(CASE WHEN is_limit_up = 1 THEN 1 END) as total_limit_days,
    AVG(change_percent) as avg_change_percent
FROM first_limit_up_stocks f
JOIN stock_daily_data d ON f.stock_code = d.stock_code
WHERE d.trade_date BETWEEN first_limit_date AND DATE_ADD(first_limit_date, INTERVAL 180 DAY)
GROUP BY f.stock_code, f.stock_name, f.first_limit_date;