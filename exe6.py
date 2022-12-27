import tkinter
import tkinter.filedialog
from PIL import Image, ImageTk
from torchvision import transforms as transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from exe3 import Car
from exe5 import detect
# 设置图片保存路径
outfile = './out_pic'

# 创建一个界面窗口
win = tkinter.Tk()
win.title("picture process")
win.geometry("1280x1080")


# 设置全局变量
original = Image.new('RGB', (300, 400))
save_img = Image.new('RGB', (300, 400))
count = 0
img2 = tkinter.Label(win)


# 实现在本地电脑选择图片
def choose_file():
    select_file = tkinter.filedialog.askopenfilename(title='选择图片')
    global strr
    strr = select_file
    print(strr)
    e.set(select_file)
    load = Image.open(select_file)
    load = transforms.Resize((300, 400))(load)
    # 声明全局变量
    global original
    original = load
    render = ImageTk.PhotoImage(load)

    img = tkinter.Label(win, image=render)
    img.image = render
    img.place(x=100, y=100)

# 中值滤波
def MedianFilter():
    temp = original
    new_im = np.asarray(temp,dtype='uint8')
    new_im = cv2.medianBlur(new_im,5)
    ren = Image.fromarray(new_im)
    render = ImageTk.PhotoImage(ren)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img
    save_img = new_im

#直方图
def histr():
    temp = original
    new_im = np.asarray(temp,dtype='uint8')
    gray = np.dot(new_im[..., :3], [0.299, 0.587, 0.114])
    gray = gray.flatten()
    wubbalubbadubdub = plt.hist(gray, bins=256)  # 绘制直方图
    plt.show()

def CarDetect():
    temp = original
    new_im = np.asarray(temp, dtype='uint8')
    pic = Car(strr)

    ren = Image.fromarray(pic)
    render = ImageTk.PhotoImage(ren)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img
    save_img = new_im

def NumDetect():
    temp = original
    new_im = np.asarray(temp, dtype='uint8')
    pic = detect(strr)

    ren = Image.fromarray(pic)
    render = ImageTk.PhotoImage(ren)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img
    save_img = new_im

#直方图均衡化
def Equalhister():
    temp = original
    new_im = np.asarray(temp,dtype='uint8')
    # 自适应均衡化
    gray = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    ren = Image.fromarray(cl1)
    render = ImageTk.PhotoImage(ren)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img
    save_img = new_im


def Canny():
    temp = original
    new_im = np.asarray(temp,dtype='uint8')
    # 自适应均衡化
    lowThreshold = 0
    max_lowThreshold = 100
    canny = cv2.Canny(new_im, lowThreshold, max_lowThreshold)

    ren = Image.fromarray(canny)
    render = ImageTk.PhotoImage(ren)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img
    save_img = new_im

def LINE(x1,x2,y1,y2):
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    return a, b, c

# Hougu变换直线检测
def Hougu():
    temp = original

    new_im = np.asarray(temp, dtype='uint8')
    print(new_im.shape)
    edges = cv2.Canny(new_im, 0, 100)
    lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 200)
    print(lines)
    if lines == None:
        print('这没直线呀')
    else:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(new_im, (x1, y1), (x2, y2), (0, 255, 0))
            a, b, c = LINE(x1,x2,y1,y2)
            print('%.2fx + %.2fy + %.2f = 0' %(a,b,c))
    # plt.figure(figsize=(10, 8), dpi=100)
    # plt.imshow(new_im), plt.title('Hougu')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    ren = Image.fromarray(new_im)
    render = ImageTk.PhotoImage(ren)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img
    save_img = new_im



# 随机水平翻转
def Horizon():
    temp = original
    new_im = transforms.RandomHorizontalFlip(p=1)(temp)
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img1
    save_img = new_im


# 随机垂直翻转
def Vertical():
    temp = original
    new_im = transforms.RandomVerticalFlip(p=1)(temp)
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img1
    save_img = new_im


# 随机角度旋转
def Rotation():
    temp = original
    new_im = transforms.RandomRotation(45)(temp)
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img1
    save_img = new_im





# 随机灰度化
def random_gray():
    temp = original
    new_im = transforms.RandomGrayscale(p=0.5)(temp)
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=800, y=100)
    global save_img
    save_img = new_im


# 设置亮度
def set_bright():
    def show_bright(ev=None):
        temp = original
        new_im = transforms.ColorJitter(brightness=scale.get())(temp)
        render = ImageTk.PhotoImage(new_im)
        global img2
        img2.destroy()
        img2 = tkinter.Label(win, image=render)
        img2.image = render
        img2.place(x=800, y=100)
        global save_img
        save_img = new_im

    top = tkinter.Tk()
    top.geometry('250x150')
    top.title('亮度设置')
    scale = tkinter.Scale(top, from_=0, to=100, orient=tkinter.HORIZONTAL, command=show_bright)
    scale.set(1)
    scale.pack()


# 设置对比度
def set_contrast():
    def show_contrast(ev=None):
        temp = original
        new_im = transforms.ColorJitter(contrast=scale.get())(temp)
        render = ImageTk.PhotoImage(new_im)
        global img2
        img2.destroy()
        img2 = tkinter.Label(win, image=render)
        img2.image = render
        img2.place(x=800, y=100)
        global save_img
        save_img = new_im

    top = tkinter.Tk()
    top.geometry('250x150')
    top.title('对比度设置')
    scale = tkinter.Scale(top, from_=0, to=100, orient=tkinter.HORIZONTAL, command=show_contrast)
    scale.set(1)
    scale.pack()


# 设置色度
def set_hue():
    def show_hue(ev=None):
        temp = original
        new_im = transforms.ColorJitter(hue=scale.get())(temp)
        render = ImageTk.PhotoImage(new_im)
        global img2
        img2.destroy()
        img2 = tkinter.Label(win, image=render)
        img2.image = render
        img2.place(x=800, y=100)
        global save_img
        save_img = new_im

    top = tkinter.Tk()
    top.geometry('250x150')
    top.title('色度设置')
    scale = tkinter.Scale(top, from_=-0.5, to=0.5, resolution=0.1, orient=tkinter.HORIZONTAL, command=show_hue)
    scale.set(1)
    scale.pack()


# 设置饱和度
def set_saturation():
    def show_saturation(ev=None):
        temp = original
        new_im = transforms.ColorJitter(saturation=scale.get())(temp)
        render = ImageTk.PhotoImage(new_im)
        global img2
        img2.destroy()
        img2 = tkinter.Label(win, image=render)
        img2.image = render
        img2.place(x=800, y=100)
        global save_img
        save_img = new_im

    top = tkinter.Tk()
    top.geometry('250x150')
    top.title('饱和度设置')
    scale = tkinter.Scale(top, from_=0, to=100, resolution=1, orient=tkinter.HORIZONTAL, command=show_saturation)
    scale.set(1)
    scale.pack()



# 保存函数
def save():
    global count
    count += 1
    save_img.save(os.path.join(outfile, 'test' + str(count) + '.jpg'))

# 显示路径
e = tkinter.StringVar()
e_entry = tkinter.Entry(win, width=68, textvariable=e)
e_entry.pack()

# 设置选择图片的按钮
button1 = tkinter.Button(win, text="选择", command=choose_file)
button1.pack()

# 设置标签分别为原图像和修改后的图像
label1 = tkinter.Label(win, text="Original Picture")
label1.place(x=200, y=50)

label2 = tkinter.Label(win, text="Modified Picture")
label2.place(x=900, y=50)

# 设置保存图片的按钮
button2 = tkinter.Button(win, text="保存", command=save)
button2.place(x=600, y=100)

# 图像去噪
button3 = tkinter.Button(win, text="中值滤波去噪", command=MedianFilter)
button3.place(x=600, y=150)

# 直方图
button4 = tkinter.Button(win, text="直方图", command=histr)
button4.place(x=600, y=200)

# 直方图均衡化
button5 = tkinter.Button(win, text="直方图均衡化", command=Equalhister)
button5.place(x=600, y=250)

# 边缘检测
button6 = tkinter.Button(win, text="边缘检测", command=Canny)
button6.place(x=600, y=300)

# 直线检测
button7 = tkinter.Button(win, text="直线检测", command=Hougu)
button7.place(x=600, y=350)

# 汽车识别
button8 = tkinter.Button(win, text="汽车识别", command=CarDetect)
button8.place(x=600, y=400)

# 数字识别
button9 = tkinter.Button(win, text="数字识别", command=NumDetect)
button9.place(x=600, y=450)

# 设置随机水平翻转按钮
button10 = tkinter.Button(win, text="随机水平翻转", command=Horizon)
button10.place(x=600, y=500)

# 设置随机垂直翻转按钮
button11 = tkinter.Button(win, text="随机垂直翻转", command=Vertical)
button11.place(x=600, y=550)

# 设置随机角度旋转按钮
button12 = tkinter.Button(win, text="随机角度旋转", command=Rotation)
button12.place(x=600, y=600)


# 设置随机灰度化按钮
button13 = tkinter.Button(win, text="设置灰度化", command=random_gray)
button13.place(x=600, y=650)

# 设置亮度的按钮
button14 = tkinter.Button(win, text="设置亮度", command=set_bright)
button14.place(x=600, y=700)

# 设置对比度的按钮
button15 = tkinter.Button(win, text="设置对比度", command=set_contrast)
button15.place(x=600, y=750)

# 设置色度按钮
button16 = tkinter.Button(win, text="设置色度", command=set_hue)
button16.place(x=600, y=800)

# 设置饱和度按钮
button17 = tkinter.Button(win, text="设置饱和度", command=set_saturation)
button17.place(x=600, y=850)



# 设置退出按钮
button0 = tkinter.Button(win, text="Exit", command=win.quit)
button0.place(x=600, y=950)
win.mainloop()
