import chardet

# byte_str = b"\344\270\252\344\272\272\350\265\204\346\226\231\345\215\241"
byte_str = b"\300\266\314\354\266\340"
byte_str_charset = chardet.detect(byte_str)  # 获取字节码编码格式

byte_str = str(byte_str, byte_str_charset.get('encoding'))  # 将八进制字节流转化为字符串
print(byte_str)

# 输出 ：
"""
个人资料卡
"""