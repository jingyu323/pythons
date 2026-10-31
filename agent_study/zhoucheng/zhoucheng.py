import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ===================== 1. 全局固定常数 =====================
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

theta = 28 * np.pi / 180  # 保持架压坡角

# ===================== 2. 读取Excel外部边界数据 =====================
def read_excel_data(file_path="lingo输入输出.xlsx"):
    """读取Excel中边界参数"""
    df = pd.read_excel(file_path, engine="openpyxl")
    data = {}
    # 读取指定列（和LINGO一一对应）
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

# 读取数据
excel_data = read_excel_data()
CSDO2 = excel_data["CSDO2"]
CSdi2 = excel_data["CSdi2"]
CST2 = excel_data["CST2"]
c1c2 = excel_data["c1c2"]
alfamin = excel_data["alfamin"] * np.pi / 180
alfamax = excel_data["alfamax"] * np.pi / 180
dwemin = excel_data["dwemin"]
dwemax = excel_data["dwemax"]
fai_max = 3.0858 * np.pi / 180

# ===================== 3. 目标函数：最小化总重量 wzong =====================
def obj_func(x):
    """
    设计变量 x 顺序：
    x[0] : alfa      接触角(rad)
    x[1] : fai       半锥角(rad)
    x[2] : dwe       滚子平均直径
    x[3] : lw        滚子长度
    x[4] : Dw        滚子大端直径
    x[5] : E
    x[6] : di2       轴承内径
    x[7] : do2       轴承外径
    x[8] : t2        轴承总宽 T
    x[9] : z         滚子数量(后续取整)
    """
    alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x

    # 基础几何赋值
    di = di2
    DO = do2
    T = t2
    B = t2
    C = t2

    lwe = lw - c1c2

    # 保持架参数
    S = 0.15 * Dw
    S1 = 6.5
    S2 = S - S1

    # 几何中间量
    D2 = E + C * np.tan(alfa) * 2
    Dwe = Dw - lw * np.tan(fai)
    Dwi = Dw - 2 * lw * np.tan(fai)
    Dpw = Dwe / np.tan(fai) * np.sin(alfa - fai)

    L = (lw + epsilon1) * np.cos(alfa - 2*fai) / np.cos(fai)
    ssss = (E - di2) / (2 * np.tan(alfa))
    a0 = -Dw * np.cos(alfa-2*fai) / (2*np.sin(fai)) + 0.5 * E / np.tan(alfa) + T
    a1 = B - a0 - L

    Si = 0.5 * Dw * np.sin(alfa-2*fai) / np.sin(fai) - L * np.tan(alfa-2*fai) - 0.5 * di
    Se = 0.5 * (DO - Dw * np.sin(alfa) / np.sin(fai))

    rou_rou = (E/(2*np.tan(alfa)) + T - a0) / np.cos(alfa - 2*fai)
    Dw_pie = (2 * rou_rou + epsilon1_bcj) * np.tan(fai)
    Dw1_pie = Dw_pie - 2 * L * np.tan(fai)

    delta_c = (Dw_pie + epsilon_delta) * np.cos(theta) - S2 * np.tan(theta)
    delta_c1 = (Dw1_pie + epsilon_delta) * np.cos(theta) - S2 * np.tan(theta)
    di_pie = 2 * rou_rou * np.sin(alfa - 2*fai)
    di_pie_pie = di_pie + (Dw_pie - Dw) * np.sin(alfa-2*fai) / np.sin(fai)

    Dcj = np.sqrt(
        (di_pie_pie + (Dw_pie + (Dw_pie+epsilon_delta)*np.sin(theta))*np.cos(alfa-fai))**2
        + ((Dw_pie - epsilon_delta)*np.cos(theta))**2
    )

    # 额定载荷
    gama = dwe * np.cos(alfa) / Dpw
    fc = 551.13373 * 0.377 * lambda_para * nu \
         * (gama ** (2/9)) * ((1-gama)**(29/27)) / ((1+gama)**(1/4)) \
         * (1 + 1.04 * (((1-gama)/(1+gama))**(143/108)))** (9/2) ** (-2/9)

    cr = bm * fc * ((lwe * np.cos(alfa))**(7/9)) * (z**(3/4)) * (dwe**(29/27)) / 1000
    c0r = 44 * (1 - dwe*np.cos(alfa)/Dpw) * z * lwe * dwe * np.cos(alfa) / 1000

    # 重量分项
    di3 = di + 2*Si + 24
    di4 = di + 2*Si
    did = Dw * np.sin(alfa - 2*fai + (0.5 + 2/di)*fai) / np.sin(fai)

    w01 = (2.042 * C * (3*DO**2 - E**2 - E*D2 - D2**2) - 10 * r3**2 * DO) * 1e-6
    w02 = (6.126 * (a0*did**2 + a1*di3**2 + L/3*(di_pie**2 + di_pie*di4 + di4**2) - di**2 * B) - 7.8 * r3**2 * di) * 1e-6
    w04 = (2.042 * lw * (Dw**2 + Dw*Dwi + Dwi**2)) * 1e-6
    w07 = (Dcj * S + 2 * S) * 1e-6 * 7.85 * (1.1 * B)

    wzong = w01 + w02 + w04 * z + w07
    return wzong

# ===================== 4. 约束条件 (scipy 标准：f(x) >= 0) =====================
def get_constraints():
    cons = []

    # 1. 额定载荷约束: cr + c0r >= 16290+50895
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
        fc = 551.13373 * 0.377 * lambda_para * nu \
             * (gama ** (2/9)) * ((1-gama)**(29/27)) / ((1+gama)**(1/4)) \
             * (1 + 1.04 * (((1-gama)/(1+gama))**(143/108)))** (9/2) ** (-2/9)
        cr = bm * fc * ((lwe * np.cos(alfa))**(7/9)) * (z**(3/4)) * (dwe**(29/27)) / 1000
        c0r = 44 * (1 - dwe*np.cos(alfa)/Dpw) * z * lwe * dwe * np.cos(alfa) / 1000
        return cr + c0r - (16290 + 50895)
    cons.append({"type": "ineq", "fun": cons_load})

    # 2. 内圈挡边厚度 gx1>0
    def cons_gx1(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        di = di2
        DO = do2
        T = t2
        B = t2
        C = t2
        L = (lw + epsilon1) * np.cos(alfa - 2*fai) / np.cos(fai)
        ssss = (E - di2) / (2 * np.tan(alfa))
        a0 = -Dw * np.cos(alfa-2*fai) / (2*np.sin(fai)) + 0.5 * E / np.tan(alfa) + T
        a1 = B - a0 - L
        gx1 = a0/B - ka0
        return gx1
    cons.append({"type": "ineq", "fun": cons_gx1})

    # 3. 内圈挡边厚度 gx2>0
    def cons_gx2(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        di = di2
        DO = do2
        T = t2
        B = t2
        C = t2
        L = (lw + epsilon1) * np.cos(alfa - 2*fai) / np.cos(fai)
        a0 = -Dw * np.cos(alfa-2*fai) / (2*np.sin(fai)) + 0.5 * E / np.tan(alfa) + T
        a1 = B - a0 - L
        gx2 = a1/B - ka1
        return gx2
    cons.append({"type": "ineq", "fun": cons_gx2})

    # 4. gx3 > 3
    def cons_gx3(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        T = t2
        C = t2
        a0 = -Dw * np.cos(alfa-2*fai) / (2*np.sin(fai)) + 0.5 * E / np.tan(alfa) + T
        gx3 = C - (T - a0 - Dw*np.sin(alfa-fai)) - r7max + c_min
        return gx3 - 3
    cons.append({"type": "ineq", "fun": cons_gx3})

    # 5. gx4 > 0
    def cons_gx4(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        T = t2
        a0 = -Dw * np.cos(alfa-2*fai) / (2*np.sin(fai)) + 0.5 * E / np.tan(alfa) + T
        gx4 = T - a0 - Dw*np.sin(alfa-fai) - lw*np.cos(alfa)/np.cos(fai) - r6max
        return gx4
    cons.append({"type": "ineq", "fun": cons_gx4})

    # 6. gx5 > 0 内外圈壁厚
    def cons_gx5(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        di = di2
        DO = do2
        T = t2
        B = t2
        C = t2
        L = (lw + epsilon1) * np.cos(alfa - 2*fai) / np.cos(fai)
        Si = 0.5 * Dw * np.sin(alfa-2*fai) / np.sin(fai) - L * np.tan(alfa-2*fai) - 0.5 * di
        gx5 = Si - 0.1*(DO - di)
        return gx5
    cons.append({"type": "ineq", "fun": cons_gx5})

    # 7. gx6 > 0
    def cons_gx6(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        di = di2
        DO = do2
        T = t2
        B = t2
        C = t2
        L = (lw + epsilon1) * np.cos(alfa - 2*fai) / np.cos(fai)
        Si = 0.5 * Dw * np.sin(alfa-2*fai) / np.sin(fai) - L * np.tan(alfa-2*fai) - 0.5 * di
        Se = 0.5 * (DO - Dw * np.sin(alfa) / np.sin(fai))
        gx6 = 0.02*(DO - di) - abs(Si - Se)
        return gx6
    cons.append({"type": "ineq", "fun": cons_gx6})

    # 8. gx7 > 0 滚子间隙
    def cons_gx7(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        J = Dw / np.sin(fai) * np.sin( np.arcsin( np.sin(alfa-fai)*np.sin(np.pi/z) ) - fai )
        gx7 = J - 0.057 * Dw
        return gx7
    cons.append({"type": "ineq", "fun": cons_gx7})

    # 9. gx8 > 0 保持架梁宽
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
        Dw1_pie = Dw_pie - 2 * L * np.tan(fai)
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

    # 10. 滚子数量上限约束
    def cons_z_max(x):
        alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
        di = di2
        DO = do2
        T = t2
        B = t2
        C = t2
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

# ===================== 5. 变量上下界 bounds =====================
def get_bounds():
    bounds = []
    # alfa: [alfamin, alfamax]
    bounds.append((alfamin, alfamax))
    # fai: [0, fai_max]
    bounds.append((1e-6, fai_max))
    # dwe: [dwemin, dwemax]
    bounds.append((dwemin, dwemax))
    # lw 滚子长度 (自定义合理范围)
    bounds.append((10, 100))
    # Dw 滚子大端直径
    bounds.append((10, 100))
    # E
    bounds.append((10, 200))
    # di2 < CSDO2
    bounds.append((10, CSdi2-1e-6))
    # do2 < CSDO2
    bounds.append((20, CSDO2-1e-6))
    # t2 < CST2
    bounds.append((10, CST2-1e-6))
    # z 滚子数量 6~60
    bounds.append((6, 60))
    return bounds

# ===================== 6. 结果写入Excel =====================
def write_result_to_excel(x, file_path="lingo输入输出.xlsx"):
    alfa, fai, dwe, lw, Dw, E, di2, do2, t2, z = x
    z_int = int(round(z))

    # 重新计算全套输出参数
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
    fc = 551.13373 * 0.377 * lambda_para * nu \
         * (gama ** (2/9)) * ((1-gama)**(29/27)) / ((1+gama)**(1/4)) \
         * (1 + 1.04 * (((1-gama)/(1+gama))**(143/108)))** (9/2) ** (-2/9)
    cr = bm * fc * ((lwe * np.cos(alfa))**(7/9)) * (z_int**(3/4)) * (dwe**(29/27)) / 1000
    c0r = 44 * (1 - dwe*np.cos(alfa)/Dpw) * z_int * lwe * dwe * np.cos(alfa) / 1000

    faidu = fai * 180 / np.pi
    alfadu = alfa * 180 / np.pi
    wzong = obj_func(x)

    out_data = {
        "cr": [cr],
        "c0r": [c0r],
        "dwe": [dwe],
        "lwe": [lwe],
        "lw": [lw],
        "fai": [faidu],
        "z": [z_int],
        "dpw": [Dpw],
        "alfa": [alfadu],
        "e": [E],
        "wzong": [wzong],
        "DO2_": [do2],
        "di2_": [di2],
        "T2_": [t2],
        "B2_": [B],
        "_C2": [C]
    }
    df_out = pd.DataFrame(out_data)
    with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_out.to_excel(writer, sheet_name="优化结果", index=False)
    print("✅ 优化结果已写入 Excel")

# ===================== 7. 主求解流程 =====================
if __name__ == "__main__":
    # 初始迭代点 (根据轴承经验手动设置一组合理初值)
    x0 = [
        (alfamin + alfamax)/2,   # alfa
        fai_max / 2,             # fai
        (dwemin + dwemax)/2,     # dwe
        40,                      # lw
        30,                      # Dw
        80,                      # E
        CSdi2 * 0.8,             # di2
        CSDO2 * 0.8,             # do2
        CST2 * 0.8,              # t2
        30                       # z
    ]

    bounds = get_bounds()
    cons = get_constraints()

    # 求解
    res = minimize(
        fun=obj_func,
        x0=x0,
        method="SLSQP",   # 支持不等式约束的求解器
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000, "disp": True}
    )

    print("\n========== 优化结果 ==========")
    print("求解状态:", res.message)
    print("是否成功:", res.success)
    print("最小总重量 wzong =", res.fun)
    print("设计变量最优解:\n", res.x)

    # 滚子数量取整
    res_x = res.x.copy()
    res_x[9] = round(res_x[9])
    print("取整后滚子数量 z =", int(res_x[9]))

    # 写入Excel
    write_result_to_excel(res_x)