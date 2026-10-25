import matplotlib.pyplot as plt
import seaborn as sns

from get_zhang_cls import StockAnalyzer


class EnhancedStockAnalyzer(StockAnalyzer):
    def __init__(self):
        super().__init__()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示

    def plot_sector_distribution(self, sector_analysis):
        """绘制板块分布图"""
        sectors = list(sector_analysis.keys())
        counts = [data['涨停数量'] for data in sector_analysis.values()]

        plt.figure(figsize=(12, 8))
        plt.bar(sectors, counts, color=sns.color_palette("husl", len(sectors)))
        plt.title('各板块涨停股票数量分布', fontsize=16)
        plt.xticks(rotation=45)
        plt.ylabel('涨停数量')
        plt.tight_layout()
        plt.show()

    def find_hot_concepts(self, stocks_data, top_n=10):
        """找出热门概念"""
        concepts = {}

        for stock in stocks_data:
            concept_desc = stock.get('概念说明', '')
            if concept_desc:
                # 简单提取关键词（实际应用中可使用更复杂的NLP技术）
                keywords = ['核聚变', '智能驾驶', '商业航天', '房地产', 'IP',
                            '医药', '零售', '食品', '光伏', '光通信']

                for keyword in keywords:
                    if keyword in concept_desc:
                        concepts[keyword] = concepts.get(keyword, 0) + 1

        sorted_concepts = sorted(concepts.items(), key=lambda x: x[1], reverse=True)
        return sorted_concepts[:top_n]


def analyze_continuous_limit_up(self, stocks_data):
    """分析连续涨停股票"""
    continuous_stocks = []

    for stock in stocks_data:
        if stock.get('涨停天数', 1) >= 3:  # 连续3天及以上涨停
            continuous_stocks.append(stock)

    return continuous_stocks


# 完整执行流程
def run_complete_analysis():
    """运行完整分析流程"""
    print("🎯 股票数据爬取与分析系统")
    print("=" * 50)

    # 基础分析
    base_analyzer = StockAnalyzer()
    stocks_data, sector_analysis = base_analyzer.main()

    if not stocks_data:
        return

    # 增强分析
    enhanced_analyzer = EnhancedStockAnalyzer()

    # 热门概念分析
    hot_concepts = enhanced_analyzer.find_hot_concepts(stocks_data)
    print(f"\n🔥 热门概念TOP10:")
    for concept, count in hot_concepts:
        print(f"  {concept}: {count}次提及")

    # 连续涨停分析
    continuous_stocks = enhanced_analyzer.analyze_continuous_limit_up(stocks_data)

    if continuous_stocks:
        print(f"\n🚀 连续涨停股票 ({len(continuous_stocks)}只):")
        for stock in continuous_stocks:
            print(f"  {stock['股票简称']} - {stock['涨停天数']}连板")

    # 可视化
    enhanced_analyzer.plot_sector_distribution(sector_analysis)


# 运行分析
if __name__ == "__main__":
    run_complete_analysis()
