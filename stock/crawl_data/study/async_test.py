import asyncio


async def my_coroutine():
    print("开始执行协程")
    await asyncio.sleep(1) # 模拟耗时操作
    print("协程执行完毕")
