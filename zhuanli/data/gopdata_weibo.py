import gopup as gp
import jupyter
import matplotlib.pyplot as plt

df_index = gp.weibo_index(word="疫情", time_type="3month")
print(df_index)



plt.figure(figsize=(15, 5))
plt.title("微博「疫情」热度走势图")
plt.xlabel("时间")
plt.ylabel("指数")
plt.plot(df_index.index, df_index['疫情'], '-', label="指数")
plt.legend()
plt.grid()
plt.show()