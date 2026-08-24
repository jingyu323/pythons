import time

print(time.time())


time_g=time.localtime(time.time())
time.strftime('%Y-%m-%d %H-%M-%S',time_g)

str_time='2018-10-1 10:11:12'
time_g=time.strptime(str_time,'%Y-%m-%d %H:%M:%S')

print(time_g)