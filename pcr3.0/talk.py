import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 设置语速
engine.setProperty('volume', 1.0)  # 设置音量
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # 设置第一个语音合成器
# voices = engine.getProperty('voices')
# for voice in voices:
#     print(voice)
engine.say("绿码")
engine.runAndWait()
engine.stop()
