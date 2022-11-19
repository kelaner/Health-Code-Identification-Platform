import pyttsx3

engine = pyttsx3.init()

engine.setProperty('rate', 150)  # 设置语速
engine.setProperty('volume', 1.0)  # 设置音量

engine.say("系统正在启动，扫描程序运行")
engine.runAndWait()
