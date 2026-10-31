import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# ===================== 全局固定常数 =====================
lambda_para = 1.0
nu = 0.83
bm = 1.1

ka0 = 0.15
ka1 = 0.13
r7max = 2.4
r6max = 2.4
c_min = 0.8

epsilon1 = 0.97
epsilon1_bcj = 0.8
epsilon_delta = 0.7
kj = 1.5
r3 = 10

theta = 28 * np.pi / 180

# 固定Excel文件路径
EXCEL_PATH = r"D:\project\agent\agent_study\zhoucheng\lingo输入输出.xlsx"

# ===================== 自动写入Excel标准表头 =====================
def init_excel_header():
    standard_headers = [
        "DO", "di", "T", "B", "CC",
        "alfamin", "alfamax", "xx", "yy",
        "_c1c2", "Dwemin", "Dwemax"
    ]
    if not os.path.exists(EXCEL_PATH):
        df_empty = pd.DataFrame(columns=standard_headers)
        df_empty.to_excel(EXCEL_PATH, engine="openpyxl", index=False)
        print(f"✅ 已创建新Excel并写入标准表头")
    else:
        df_exist = pd.read_excel(EXCEL_PATH, engine="openpyxl")
        if list(df_exist.columns) != standard_headers:
            df_new = pd.DataFrame(columns=standard_headers)
            for i, col in enumerate(standard_headers):
                if i < len(df_exist.columns):
                    df_new[col] = df_exist.iloc[:, i]
            df_new.to_excel(EXCEL_PATH, engine="openpyxl", index=False)
            print(f"✅ 已修正Excel表头")
        else:
            print("✅ Excel表头已匹配")

# ===================== 读取Excel边界数据（增加空表判断） =====================
def read_excel_data():
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    print(f"📊 读取到的Excel列名：{list(df.columns)}")

    # 判断是否存在数据行
    if df.shape[0] == 0:
        raise Exception("❌ 错误：Excel仅有表头，无参数数据！\n请在Excel第一行填入12列对应数值后重新运行。")

    print(f"📊 读取到的边界数据：{df.iloc[0].tolist()}")
    data = {}
    data["CSDO2"] = df["DO"].iloc[0]
    data["CSdi2"] = df["di"].iloc[0]
    data["CST2"] = df["T"].iloc[0]
    data["CSB2"] = df["B"].iloc[0]
    data["CSC2"] = df["CC"].iloc[0]
    data["alfamin"] = df["alfamin"].iloc[0]
    data["alfamax"] = df["alfamax"].iloc[0]
    data["xx"] = df["xx"].iloc[0]
    data["yy"] = df["yy"].iloc[0]
    data["c1c2"] = df["_c1c2"].iloc[0]
    data["dwemin"] = df["Dwemin"].iloc[0]
    data["dwemax"] = df["Dwemax"].iloc[0]
    return data

# ===================== 目标函数 =====================
def obj_func(x):
    alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
    di = di2
    DO = do2
    T = t2
    B = t2
    C = t2
    lwe = lw - c1c2

    S = 0.15 * Dw
    S1 = 6.5
    S2 = S - S1

    D2 = E + C * np.tan(alfa) * 2
    Dwe = Dw - lw * np.tan(fai)
    Dwi = Dw - 2 * lw * np.tan(fai)
    Dpw = Dwe / np.tan(fai) * np.sin(alfa - fai)

    L = (lw + epsilon1) * np.cos(alfa - 2*fai) / np.cos(fai)
    a0 = -Dw * np.cos(alfa-2*fai) / (2*np.sin(fai)) + 0.5 * E / np.tan(alfa) + T
    a1 = B - a0 - L

    Si = 0.5 * Dw * np.sin(alfa-2*fai) / np.sin(fai) - L * np.tan(alfa-2*fai) - 0.5 * di
    Se = 0.5 * (DO - Dw * np.sin(alfa) / np.sin(fai))

    rou_rou = (E/(2*np.tan(alfa)) + T - a0) / np.cos(alfa - 2*fai)
    Dw_pie = (2 * rou_rou + epsilon1_bcj) * np.tan(fai)
    Dw1_pie = Dw_pie - 2 * L * np.tan(fai)

    delta_c = (Dw_pie + epsilon_delta) * np.cos(theta) - S2 * np.tan(theta)
    di_pie = 2 * rou_rou * np.sin(alfa - 2*fai)
    di_pie_pie = di_pie + (Dw_pie - Dw) * np.sin(alfa-2*fai) / np.sin(fai)

    Dcj = np.sqrt(
        (di_pie_pie + (Dw_pie + (Dw_pie+epsilon_delta)*np.sin(theta))*np.cos(alfa-fai))**2
        + ((Dw_pie - epsilon_delta)*np.cos(theta))**2
    )

    gama = dwe * np.cos(alfa) / Dpw
    term_inner = (1 + 1.04 * (((1 - gama) / (1 + gama)) ** (143/108)))
    fc = 551.13373 * 0.377 * lambda_para * nu \
         * (gama ** (2/9)) * ((1-gama)**(29/27)) / ((1+gama)**(1/4)) \
         * (term_inner ** (9/2)) ** (-2/9)

    cr = bm * fc * ((lwe * np.cos(alfa))**(7/9)) * (z**(3/4)) * (dwe**(29/27)) / 1000
    c0r = 44 * (1 - dwe*np.cos(alfa)/Dpw) * z * lwe * dwe * np.cos(alfa) / 1000

    di3 = di + 2*Si + 24
    di4 = di + 2*Si
    did = Dw * np.sin(alfa - 2*fai + (0.5 + 2/di)*fai) / np.sin(fai)

    w01 = (2.042 * C * (3*DO**2 - E**2 - E*D2 - D2**2) - 10 * r3**2 * DO) * 1e-6
    w02 = (6.126 * (a0*did**2 + a1*di3**2 + L/3*(di_pie**2 + di_pie*di4 + di4**2) - di**2 * B) - 7.8 * r3**2 * di) * 1e-6
    w04 = (2.042 * lw * (Dw**2 + Dw*Dwi + Dwi**2)) * 1e-6
    w07 = (Dcj * S + 2 * S) * 1e-6 * 7.85 * (1.1 * B)

    wzong = w01 + w02 + w04 * z + w07
    return wzong

# ===================== 约束条件 =====================
def get_constraints():
    cons = []
    def cons_load(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        di = di2
        DO = do2
        T = t2
        B = t2
        C = t2
        lwe = lw - c1c2
        S = 0.15 * Dw
        S1 = 6.5
        S2 = S - S1
        D2 = E + C * np.tan(alfa) * 2
        Dwe = Dw - lw * np.tan(fai)
        Dwi = Dw - 2 * lw * np.tan(fai)
        Dpw = Dwe / np.tan(fai) * np.sin(alfa - fai)
        gama = dwe * np.cos(alfa) / Dpw
        term_inner = (1 + 1.04 * (((1 - gama) / (1 + gama)) ** (143/108)))
        fc = 551.13373 * 0.377 * lambda_para * nu \
             * (gama ** (2/9)) * ((1-gama)**(29/27)) / ((1+gama)**(1/4)) \
             * (term_inner ** (9/2)) ** (-2/9)
        cr = bm * fc * ((lwe * np.cos(alfa))**(7/9)) * (z**(3/4)) * (dwe**(29/27)) / 1000
        c0r = 44 * (1 - dwe*np.cos(alfa)/Dpw) * z * lwe * dwe * np.cos(alfa) / 1000
        return cr + c0r - (16290 + 50895)
    cons.append({"type": "ineq", "fun": cons_load})

    def cons_gx1(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        B = t2
        L = (lw + epsilon1) * np.cos(alfa - 2*fai) / np.cos(fai)
        a0 = -Dw * np.cos(alfa-2*fai) / (2*np.sin(fai)) + 0.5 * E / np.tan(alfa) + t2
        gx1 = a0/B - ka0
        return gx1
    cons.append({"type": "ineq", "fun": cons_gx1})

    def cons_gx2(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        B = t2
        L = (lw + epsilon1) * np.cos(alfa - 2*fai) / np.cos(fai)
        a0 = -Dw * np.cos(alfa-2*fai) / (2*np.sin(fai)) + 0.5 * E / np.tan(alfa) + t2
        a1 = B - a0 - L
        gx2 = a1/B - ka1
        return gx2
    cons.append({"type": "ineq", "fun": cons_gx2})

    def cons_gx3(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        T = t2
        C = t2
        a0 = -Dw * np.cos(alfa-2*fai) / (2*np.sin(fai)) + 0.5 * E / np.tan(alfa) + T
        gx3 = C - (T - a0 - Dw*np.sin(alfa-fai)) - r7max + c_min
        return gx3 - 3
    cons.append({"type": "ineq", "fun": cons_gx3})

    def cons_gx4(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        T = t2
        a0 = -Dw * np.cos(alfa-2*fai) / (2*np.sin(fai)) + 0.5 * E / np.tan(alfa) + T
        gx4 = T - a0 - Dw*np.sin(alfa-fai) - lw*np.cos(alfa)/np.cos(fai) - r6max
        return gx4
    cons.append({"type": "ineq", "fun": cons_gx4})

    def cons_gx5(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        di = di2
        DO = do2
        L = (lw + epsilon1) * np.cos(alfa - 2*fai) / np.cos(fai)
        Si = 0.5 * Dw * np.sin(alfa-2*fai) / np.sin(fai) - L * np.tan(alfa-2*fai) - 0.5 * di
        gx5 = Si - 0.1*(DO - di)
        return gx5
    cons.append({"type": "ineq", "fun": cons_gx5})

    def cons_gx6(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        di = di2
        DO = do2
        L = (lw + epsilon1) * np.cos(alfa - 2*fai) / np.cos(fai)
        Si = 0.5 * Dw * np.sin(alfa-2*fai) / np.sin(fai) - L * np.tan(alfa-2*fai) - 0.5 * di
        Se = 0.5 * (DO - Dw * np.sin(alfa) / np.sin(fai))
        gx6 = 0.02*(DO - di) - abs(Si - Se)
        return gx6
    cons.append({"type": "ineq", "fun": cons_gx6})

    def cons_gx7(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        J = Dw / np.sin(fai) * np.sin( np.arcsin( np.sin(alfa-fai)*np.sin(np.pi/z) ) - fai )
        gx7 = J - 0.057 * Dw
        return gx7
    cons.append({"type": "ineq", "fun": cons_gx7})

    def cons_gx8(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        di = di2
        DO = do2
        T = t2
        B = t2
        C = t2
        lwe = lw - c1c2
        S = 0.15 * Dw
        S1 = 6.5
        S2 = S - S1
        D2 = E + C * np.tan(alfa) * 2
        Dwe = Dw - lw * np.tan(fai)
        Dwi = Dw - 2 * lw * np.tan(fai)
        Dpw = Dwe / np.tan(fai) * np.sin(alfa - fai)
        L = (lw + epsilon1) * np.cos(alfa - 2*fai) / np.cos(fai)
        a0 = -Dw * np.cos(alfa-2*fai) / (2*np.sin(fai)) + 0.5 * E / np.tan(alfa) + T
        a1 = B - a0 - L
        rou_rou = (E/(2*np.tan(alfa)) + T - a0) / np.cos(alfa - 2*fai)
        Dw_pie = (2 * rou_rou + epsilon1_bcj) * np.tan(fai)
        delta_c = (Dw_pie + epsilon_delta) * np.cos(theta) - S2 * np.tan(theta)
        di_pie = 2 * rou_rou * np.sin(alfa - 2*fai)
        di_pie_pie = di_pie + (Dw_pie - Dw) * np.sin(alfa-2*fai) / np.sin(fai)
        Dcj = np.sqrt(
            (di_pie_pie + (Dw_pie + (Dw_pie+epsilon_delta)*np.sin(theta))*np.cos(alfa-fai))**2
            + ((Dw_pie - epsilon_delta)*np.cos(theta))**2
        )
        term = delta_c / (Dcj + (2*S1+S2)*np.cos(theta))
        Cb = (Dcj + (2*S1+S2)*np.cos(theta)) * np.sin( np.pi/z - np.arcsin(term) )
        gx8 = Cb - kj * S
        return gx8
    cons.append({"type": "ineq", "fun": cons_gx8})

    def cons_z_max(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        T = t2
        S = 0.15 * Dw
        S1 = 6.5
        S2 = S - S1
        L = (lw + epsilon1) * np.cos(alfa - 2*fai) / np.cos(fai)
        a0 = -Dw * np.cos(alfa-2*fai) / (2*np.sin(fai)) + 0.5 * E / np.tan(alfa) + T
        rou_rou = (E/(2*np.tan(alfa)) + T - a0) / np.cos(alfa - 2*fai)
        Dw_pie = (2 * rou_rou + epsilon1_bcj) * np.tan(fai)
        delta_c = (Dw_pie + epsilon_delta) * np.cos(theta) - S2 * np.tan(theta)
        di_pie = 2 * rou_rou * np.sin(alfa - 2*fai)
        di_pie_pie = di_pie + (Dw_pie - Dw) * np.sin(alfa-2*fai) / np.sin(fai)
        Dcj = np.sqrt(
            (di_pie_pie + (Dw_pie + (Dw_pie+epsilon_delta)*np.sin(theta))*np.cos(alfa-fai))**2
            + ((Dw_pie - epsilon_delta)*np.cos(theta))**2
        )
        denom = Dcj + (2*S1+S2)*np.cos(theta)
        t1 = np.arcsin(delta_c / denom)
        t2 = np.arcsin(kj * S / denom)
        z_upper = np.pi / (t1 + t2)
        return z_upper - z
    cons.append({"type": "ineq", "fun": cons_z_max})
    return cons

# ===================== 变量上下界 =====================
def get_bounds():
    bounds = [
        (alfamin, alfamax),
        (1e-6, fai_max),
        (dwemin, dwemax),
        (10, 100),
        (10, 100),
        (10, 200),
        (10, CSdi2-1e-6),
        (20, CSDO2-1e-6),
        (10, CST2-1e-6),
        (6, 60)
    ]
    return bounds

# ===================== 优化结果写入Excel =====================
def write_result_to_excel(x):
    alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
    z_int = int(round(z))
    di = di2
    DO = do2
    T = t2
    B = t2
    C = t2
    lwe = lw - c1c2
    S = 0.15 * Dw
    S1 = 6.5
    S2 = S - S1
    D2 = E + C * np.tan(alfa) * 2
    Dwe = Dw - lw * np.tan(fai)
    Dwi = Dw - 2 * lw * np.tan(fai)
    Dpw = Dwe / np.tan(fai) * np.sin(alfa - fai)

    gama = dwe * np.cos(alfa) / Dpw
    term_inner = (1 + 1.04 * (((1 - gama) / (1 + gama)) ** (143/108)))
    fc = 551.13373 * 0.377 * lambda_para * nu \
         * (gama ** (2/9)) * ((1-gama)**(29/27)) / ((1+gama)**(1/4)) \
         * (term_inner ** (9/2)) ** (-2/9)
    cr = bm * fc * ((lwe * np.cos(alfa))**(7/9)) * (z_int**(3/4)) * (dwe**(29/27)) / 1000
    c0r = 44 * (1 - dwe*np.cos(alfa)/Dpw) * z_int * lwe * dwe * np.cos(alfa) / 1000

    faidu = fai * 180 / np.pi
    alfadu = alfa * 180 / np.pi
    wzong = obj_func(x)

    out_data = {
        "cr": [cr], "c0r": [c0r], "dwe": [dwe], "lwe": [lwe], "lw": [lw],
        "fai": [faidu], "z": [z_int], "dpw": [Dpw], "alfa": [alfadu],
        "e": [E], "wzong": [wzong], "DO2_": [do2], "di2_": [di2],
        "T2_": [t2], "B2_": [B], "_C2": [C]
    }
    df_out = pd.DataFrame(out_data)
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_out.to_excel(writer, sheet_name="优化结果", index=False)
    print("✅ 优化结果已写入Excel【优化结果】工作表")
    return out_data

# ===================== 可视化结果 =====================
def plot_visual(x0, x_opt, res, excel_out):
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    alfa0, fai0, dwe0, lw0, Dw0, E0, di2_0, do2_0, t2_0, z0 = x0
    alfa_opt, fai_opt, dwe_opt, lw_opt, Dw_opt, E_opt, di2_opt, do2_opt, t2_opt, z_opt = x_opt
    z_int = int(round(z_opt))

    fig = plt.figure(figsize=(16, 12))
    ax1 = plt.subplot(2, 3, 1)
    weight = [obj_func(x0), res.fun]
    lab = ["初始值", "最优值"]
    ax1.bar(lab, weight, color=["#7393B3", "#2E8B57"])
    ax1.set_title("轴承总重量对比", fontsize=12)
    ax1.set_ylabel("总重量")
    for i, v in enumerate(weight):
        ax1.text(i, v+0.01, f"{v:.4f}", ha="center")

    ax2 = plt.subplot(2, 3, 2)
    load_name = ["额定动载荷cr", "额定静载荷c0r"]
    load_val = [excel_out["cr"][0], excel_out["c0r"][0]]
    ax2.bar(load_name, load_val, color="#FF7F50")
    ax2.set_title("额定载荷", fontsize=12)
    for i, v in enumerate(load_val):
        ax2.text(i, v+200, f"{v:.1f}", ha="center")

    ax3 = plt.subplot(2, 3, 3)
    ang_name = ["接触角α", "半锥角φ"]
    ang_opt = [excel_out["alfa"][0], excel_out["fai"][0]]
    ax3.bar(ang_name, ang_opt, color="#6A5ACD")
    ax3.set_title("优化后角度(°)", fontsize=12)
    for i, v in enumerate(ang_opt):
        ax3.text(i, v+0.2, f"{v:.2f}", ha="center")

    ax4 = plt.subplot(2, 3, 4)
    dim_name = ["内径di2", "外径do2", "总宽T2", "滚子长lw"]
    dim0 = [di2_0, do2_0, t2_0, lw0]
    dim_opt = [di2_opt, do2_opt, t2_opt, lw_opt]
    x = np.arange(len(dim_name))
    width = 0.35
    ax4.bar(x-width/2, dim0, width, label="初始值", color="#87CEEB")
    ax4.bar(x+width/2, dim_opt, width, label="最优值", color="#3CB371")
    ax4.set_xticks(x)
    ax4.set_xticklabels(dim_name)
    ax4.set_title("关键尺寸对比", fontsize=12)
    ax4.legend()

    ax5 = plt.subplot(2, 3, 5)
    ax5.text(0.5, 0.6, f"优化后滚子数量\nZ = {z_int}", fontsize=16, ha="center")
    ax5.set_xlim(0,1)
    ax5.set_ylim(0,1)
    ax5.axis("off")
    ax5.set_title("滚子数量", fontsize=12)

    ax6 = plt.subplot(2, 3, 6)
    def calc_cons_val(x, fun):
        return fun(x)
    cons_list = get_constraints()
    cons_names = ["载荷余量", "挡边1余量", "挡边2余量", "端面间隙gx3", "壁厚gx5", "滚子间隙gx7"]
    cons_vals = [
        calc_cons_val(x_opt, cons_list[0]["fun"]),
        calc_cons_val(x_opt, cons_list[1]["fun"]),
        calc_cons_val(x_opt, cons_list[2]["fun"]),
        calc_cons_val(x_opt, cons_list[3]["fun"]),
        calc_cons_val(x_opt, cons_list[5]["fun"]),
        calc_cons_val(x_opt, cons_list[6]["fun"])
    ]
    ax6.bar(range(len(cons_names)), cons_vals, color="#CD853F")
    ax6.set_xticks(range(len(cons_names)))
    ax6.set_xticklabels(cons_names, rotation=45, ha="right")
    ax6.set_title("约束余量校验(>0为合格)", fontsize=12)
    ax6.axhline(y=0, color="r", linestyle="--")

    plt.tight_layout()
    plt.savefig("轴承优化结果图.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("✅ 可视化图片已保存：轴承优化结果图.png")

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 1. 初始化表头
    init_excel_header()

    # 2. 读取数据（带空表校验）
    try:
        excel_data = read_excel_data()
    except Exception as e:
        print(e)
        input("请补充Excel数据后按回车退出...")
        exit()

    # 赋值全局参数
    CSDO2 = excel_data["CSDO2"]
    CSdi2 = excel_data["CSdi2"]
    CST2 = excel_data["CST2"]
    c1c2 = excel_data["c1c2"]
    alfamin = excel_data["alfamin"] * np.pi / 180
    alfamax = excel_data["alfamax"] * np.pi / 180
    dwemin = excel_data["dwemin"]
    dwemax = excel_data["dwemax"]
    fai_max = 3.0858 * np.pi / 180

    # 初始迭代点
    x0 = [
        (alfamin + alfamax)/2,
        fai_max / 2,
        (dwemin + dwemax)/2,
        40, 30, 80,
        CSdi2 * 0.8,
        CSDO2 * 0.8,
        CST2 * 0.8,
        30
    ]

    bounds = get_bounds()
    cons = get_constraints()

    # 开始优化
    print("\n🚀 开始优化求解...")
    res = minimize(
        fun=obj_func,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000, "disp": True}
    )

    # 输出结果
    print("\n========== 优化结果汇总 ==========")
    print("求解状态:", res.message)
    print("是否收敛:", res.success)
    print("最小总重量 wzong =", round(res.fun, 4))
    print("原始最优变量:\n", np.round(res.x, 4))

    res_x = res.x.copy()
    res_x[9] = round(res_x[9])
    print("取整后滚子数量 z =", int(res_x[9]))

    # 写入+绘图
    excel_result = write_result_to_excel(res_x)
    plot_visual(x0, res_x, res, excel_result)

    print("\n🎉 全部流程执行完毕！")