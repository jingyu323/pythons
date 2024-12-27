import requests
import os


class GetTsVideo():
    def __init__(self, m3u8_name, base_url, saved_path, savedVideoName):
        self.m3u8_name = m3u8_name
        self.base_url = base_url
        self.saved_path = saved_path
        self.savedVideoName = savedVideoName

    def getTsList(self):
        """
        根据给出的m3u8文件和基础url地址，得到ts名称列表
        """
        response = requests.get(self.base_url + "/" + self.m3u8_name, stream=True, verify=False)
        rslist = response.text.split('\n')


    def downloadTsFile(self):
        """
        根据上面得到的ts名称列表，结合基础url获得ts文件真正的请求url，并保存到本地,
        每保存一个，就打印一下提示信息"保存完毕"
        """
        for i in self.tslist:
            tsURL = self.base_url + "/" + i
            try:
                response = requests.get(tsURL, stream=True, verify=False)
            except Exception as e:
                print("异常请求：%s" % e.args)
                return
            tsSavedPath = self.saved_path + "/" + i
            with open(tsSavedPath, "wb+") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print(f"TS文件:{tsURL}下载完毕！！")

    def mergeTsFile(self):
        """
        可以通过使用命令行命令（或者命令行调用ffmpeg）
        也可以直接将视频文件都粗暴写入到一个mp4文件中（不考虑文件格式等问题）
        """
        savedFile = os.path.join(self.saved_path, self.savedVideoName) + ".mp4"
        with open(savedFile, 'wb+') as f:
            for i in self.tslist:
                ts_video_path = os.path.join(self.saved_path, i)
                f.write(open(ts_video_path, 'rb').read())
                print(f"写入{i}文件结束")



def get_ts_urls(m3u8_path, base_url):
    urls = []
    with open(m3u8_path,"r") as file:

        lines = file.readlines()
        for line in lines:
            # print(line)
    
            if line.find(".ts")>0:
               urls.append(base_url + line.strip("\n"))

    return urls

if __name__ == '__main__':

    urls=  get_ts_urls("url","")
    print(urls)

    m3u8file = ""
    # m3u8file = "16978848c8546fb7bfe6156e6da07016_1280x720_1720000.m3u8"
    # 给出最直接包含ts文件名的m3u8文件的文件名
    baseurl = "14920d41&v=3&time=0"
    # 给出当前这个视频的m3u8和ts前缀rul
    savepath = "./"
    # 给出要保存ts文件的位置
    savedVideoName = "rs"
    # 给出要保存的合并后的mp4文件的名称

    obget = GetTsVideo(m3u8_name=m3u8file, base_url=baseurl, saved_path=savepath, savedVideoName=savedVideoName)
    tslist = obget.getTsList()
    obget.downloadTsFile()
    obget.mergeTsFile()
    """
    1. 从m3u8文件中获取ts文件列表
    2. 结合基础url和ts文件列表下载ts文件到savepath文件夹中
    3. 将savepath文件夹中的ts文件合并，写入savedVideoName文件中，同时还是保存在savepath文件夹中
    """
