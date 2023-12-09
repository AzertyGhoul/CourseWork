from tkinter import *
from tkinter import font
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from tkinter import PhotoImage
from PIL import Image, ImageTk

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

import numpy as np
import cv2
import os


WINDOW = Tk()
WIDTH = 1366
HEIGHT = 768
IMAGE = ""
MODEL = ""

class Colors:
    primary = "#110E10"
    font = "#FFFFFF"

def close():
    WINDOW.destroy()
    WINDOW.quit()

def loadImageGui():
    loadImage = PhotoImage(file = os.getcwd() + "/GUI/load_image.png")
    buttonLoadModel = PhotoImage(file = os.getcwd() + "/GUI/import_model.png")
    buttonExportResult = PhotoImage(file = os.getcwd() + "/GUI/save_image.png")
    shadow = PhotoImage(file = os.getcwd()  + "/GUI/shadow.png")

    return[loadImage, buttonLoadModel, buttonExportResult, shadow]

def loadImage(labelImg, info):
    global MODEL
    global finaly
    global IMAGE

    file_name = fd.askopenfilename(
        filetypes = [(["PNG", "JPG"], [".png", ".jpg"])],
        initialdir = os.getcwd()
    )

    if(file_name != ""):
        labalImage = load_img(file_name)
        labalImage = np.array(labalImage, dtype = np.uint8)
        helpImage = load_img(file_name, target_size = (224, 224))
        arrImage = img_to_array(helpImage) / 255.0 

        h,w,d = labalImage.shape

        test_arr = arrImage.reshape(1, 224, 224, 3)
        coords = MODEL.predict(test_arr)

        denorm = np.array([w,w,h,h])
        coords = coords * denorm
        coords = coords.astype(np.int32)

        xmin, xmax, ymin, ymax = coords[0]
        pt1 =(xmin, ymin)
        pt2 =(xmax, ymax)
        cv2.rectangle(labalImage,pt1,pt2,(0, 255, 0),3)

        returnToImage = Image.fromarray(labalImage)
        IMAGE = returnToImage
        finaly = ImageTk.PhotoImage(returnToImage.resize((1275, 482)))

        labelImg.configure(image = finaly)
        info.configure(text = "Для последующей работы снова загрузите изображениe")
    elif (MODEL == ""):
        mb.showerror("Ошибка", "Вы не загрузили нейронную сеть")
    else:
        mb.showerror("Ошибка", "Вы не указали путь к файлу")

def importModel():
    global MODEL
    try:
        file_name = fd.askopenfilename(
            filetypes = [("HDF5 File", ".h5")],
            initialdir = os.getcwd()
        )
        MODEL = load_model(file_name)
    except OSError:
        mb.showerror("Ошибка", "Вы не указали путь к файлу")

def saveImage():
        file_name = fd.asksaveasfilename(
            initialdir = os.getcwd(),
            filetypes = [("PNG File", "*.png")]
        )

        if(IMAGE != ""):
            IMAGE.save(file_name + ".png", format = "png")
            IMAGE.close
        else:
            mb.showerror("Ошибка", "Нейронная сеть еще не отработала")

def windowInit(buttonLoadI, buttonExportResultI, buttonLoadModelI, shadow):
    WINDOW.title("Course work")

    fontForText = font.Font(family = "Poppins", size = 15, weight = "bold")

    frame = Frame(WINDOW, width = WIDTH, height = HEIGHT, background = Colors.primary)
    frame.pack(side = TOP, pady = 50)

    info = Label(WINDOW, text = "Для работы загрузите модель NN, а затем изображение", font = fontForText, background = Colors.primary, fg = Colors.font)
    info.pack(side = BOTTOM, pady = 52)

    labelImg = Label(WINDOW, image = shadow, bg = Colors.primary, width = 1275, height = 482)
    labelImg.place(x = 45, y = 149)

    buttonLoad = Button(frame, image = buttonLoadI, command = lambda: loadImage(labelImg, info), borderwidth = 0, activebackground = Colors.primary, bg = Colors.primary)
    buttonLoad.pack(side = LEFT, padx = 48)

    buttonExportResult = Button(frame, image = buttonExportResultI, command = saveImage, borderwidth = 0, activebackground = Colors.primary, bg = Colors.primary)
    buttonExportResult.pack(side = RIGHT, padx = 48)

    buttonLoadModel = Button(frame, image = buttonLoadModelI, command = lambda: importModel(), borderwidth = 0, activebackground = Colors.primary, bg = Colors.primary)
    buttonLoadModel.pack(side = TOP)

    WINDOW.geometry('%dx%d+%d+%d' %  (WIDTH, HEIGHT, 
    WINDOW.winfo_screenwidth() / 2 - WIDTH / 2,
    WINDOW.winfo_screenheight() / 2 - HEIGHT / 2))

    WINDOW.configure(background = Colors.primary)

    WINDOW.resizable(width = False, height = False)
    WINDOW.protocol('WM_DELETE_WINDOW', close)

def main():

    button_load = loadImageGui()[0]
    buttonLoadModel = loadImageGui()[1]
    buttonExportResult = loadImageGui()[2]
    shadow = loadImageGui()[3]

    windowInit(button_load, buttonExportResult, buttonLoadModel, shadow)
    WINDOW.mainloop()

if __name__ == "__main__":
    main()


