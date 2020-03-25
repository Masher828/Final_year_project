import threading
import tkinter.filedialog
from tkinter import *
import numpy as np
from PIL import ImageTk, Image
from tkinter import ttk

from botocore import args

sys.path.insert(1, "Face_Recognition/")
sys.path.insert(2, "Emoji_predictor/")
sys.path.insert(3, "Image_Classification/")
sys.path.insert(4, "odd_one_out/")
sys.path.insert(5, "Word_analogy/")

import Face_Record, Face_Recognition, emojii, Flowers, odd_one_out, word_analogy


def word_analog(old_frame):
    old_frame.destroy()
    frame = Frame(root)
    frame.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)
    Button(frame, image=cross, command=close).place(relx=0.982, rely=0, relwidth=0.02, relheight=0.03)
    Button(frame, image=mini, command=minimize).place(relx=0.962, rely=0, relwidth=0.02, relheight=0.03)
    word_a = Label(frame, text="Enter word A", fg="white", bg="#013A55",
                   font='Helvetica 10 bold')
    word_a.place(relx=0.1, rely=0.1, relwidth=0.27, relheight=0.09)
    a = Entry(frame)
    a.place(relx=0.4, rely=0.1, relwidth=0.27, relheight=0.09)
    word_b = Label(frame, text="Enter word B", fg="white", bg="#013A55",
                   font='Helvetica 10 bold')
    word_b.place(relx=0.1, rely=0.3, relwidth=0.27, relheight=0.09)
    b = Entry(frame)
    b.place(relx=0.4, rely=0.3, relwidth=0.27, relheight=0.1)
    word_c = Label(frame, text="Enter word C", fg="white", bg="#013A55",
                   font='Helvetica 10 bold')
    word_c.place(relx=0.1, rely=0.6, relwidth=0.37, relheight=0.09)
    c = Entry(frame)
    c.place(relx=0.4, rely=0.6, relwidth=0.2, relheight=0.1)
    submit = Button(root, text="Predict Word", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                    command=lambda: predict_word(frame, [a.get(), b.get(), c.get()]))
    submit.place(relx=0.7, rely=0.7, relwidth=0.2, relheight=0.1)
    main_menu = Button(root, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                       command=lambda: Menu([frame]))
    main_menu.place(relx=0.29, rely=0.933, relwidth=0.24, relheight=0.05)


def predict_word(frame, words):
    odd = word_analogy.word_analogyy(glove_dictionary, words)
    Button(frame, text=odd, command=lambda: Menu([frame, frame1])).place(relx=0.7, rely=0.5, relwidth=0.2,
                                                                         relheight=0.2)


def odd_out(old_frame):
    old_frame.destroy()
    frame = Frame(root)
    frame.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)
    Button(frame, image=cross, command=close).place(relx=0.982, rely=0, relwidth=0.02, relheight=0.03)
    Button(frame, image=mini, command=minimize).place(relx=0.962, rely=0, relwidth=0.02, relheight=0.03)
    label = Label(frame, text="Enter the words seprated by comma (,) ", fg="white", bg="#013A55",
                  font='Helvetica 10 bold')
    label.place(relx=0.1, rely=0.1, relwidth=0.37, relheight=0.09)
    sent = Entry(frame)
    sent.place(relx=0.1, rely=0.3, relwidth=0.2, relheight=0.1)
    submit = Button(frame, text="Find odd word ", command=lambda: odd_name(glove_dictionary, sent.get(), frame))
    submit.place(relx=0.15, rely=0.43, relwidth=0.1, relheight=0.1)
    main_menu = Button(root, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                       command=lambda: Menu([frame]))
    main_menu.place(relx=0.29, rely=0.933, relwidth=0.24, relheight=0.05)


def odd_name(glove, sent, frame):
    odd = odd_one_out.externalodd_out(glove, sent)
    Button(frame, text=odd, command=lambda: Menu([frame, frame1])).place(relx=0.5, rely=0.5, relwidth=0.2,
                                                                         relheight=0.2)


def flower_recognition(frame1):
    frame1.destroy()
    frame = Frame(root)
    frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    Button(frame, image=cross, command=close).place(relx=0.982, rely=0, relwidth=0.02, relheight=0.03)
    Button(frame, image=mini, command=minimize).place(relx=0.962, rely=0, relwidth=0.02, relheight=0.03)
    filename_path = tkinter.filedialog.askopenfilename()
    label = Label(frame, text=filename_path, fg="white", bg="#013A55", font='Helvetica 10 bold')
    label.place(relx=0.06, rely=0.06, relwidth=0.37, relheight=0.09)

    submit = Button(frame, text="Recognize", bg="#013A55", font="Helvetica 18 bold",
                    command=lambda: recog_flower_name(filename_path, frame))
    submit.place(relx=0.31, rely=0.19, relwidth=0.1, relheight=0.1)
    main_menu = Button(frame, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                  command=lambda: Menu([frame]))
    main_menu.place(relx=0.29, rely=0.933, relwidth=0.24, relheight=0.05)


def recog_flower_name(filename_path, frame1):
    path = 'Files/Image_Classification/Output/'
    flower_name = Flowers.flower_recog(filename_path)
    img = PhotoImage(file=path + flower_name + ".png")
    data = open(path + flower_name + ".txt", encoding="utf8")
    contents = data.read()

    frame1.destroy()
    frame = Frame(root)
    frame.place(relx=0.02, rely=0.02, relwidth=1, relheight=1)
    Label(frame, text=contents, fg="black", bg="#013A55", font='Helvetica 10').place(relx=0.0, rely=0.3)
    Button(frame, image=img, command=lambda: Menu([frame])).place(relx=0.35, rely=0.0, relheight=0.3)
    Button(root, image=cross, command=close).place(relx=0.982, rely=0, relwidth=0.02, relheight=0.03)
    Button(root, image=mini, command=minimize).place(relx=0.962, rely=0, relwidth=0.02, relheight=0.03)
    next = Button(frame, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                  command=lambda: Menu([frame]))
    next.place(relx=0.29, rely=0.933, relwidth=0.24, relheight=0.05)
    root.mainloop()


def emojipredict(old_frame):
    old_frame.destroy()
    frame = Frame(root)
    frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    Button(frame, image=cross, command=close).place(relx=0.982, rely=0, relwidth=0.02, relheight=0.03)
    Button(frame, image=mini, command=minimize).place(relx=0.962, rely=0, relwidth=0.02, relheight=0.03)
    label = Label(frame, text="Enter the text here ", fg="#738f93", font='Helvetica 22 bold')
    label.place(relx=0.01, rely=0.1, relwidth=0.2, relheight=0.1)
    sent = Entry(frame)
    sent.place(relx=0.25, rely=0.1, relwidth=0.2, relheight=0.1)
    submit = Button(frame, text="Get Emoji", bg="#013A55", font="Helvetica 18 bold",
                    command=lambda: emojipredict_image(glove_dictionary, sent.get(), frame))
    submit.place(relx=0.30, rely=0.23, relwidth=0.1, relheight=0.1)
    Button(frame, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
           command=lambda: Menu([frame])).place(relx=0.29, rely=0.933, relwidth=0.24, relheight=0.05)


def emojipredict_image(glove_dicitionary, sent, frame):
    path = emojii.pred(glove_dicitionary, sent)
    photo = PhotoImage(file=path)

    Button(frame, image=photo, command=lambda: Menu([frame])).place(relx=0.1, rely=0.4)
    Label(frame, text="Click on Image to close").place(relx=0.034, rely=0.87, relwidth=0.3, relheight=0.1)
    root.mainloop()


def newFaceRecord(old_frame):
    old_frame.destroy()
    frame = Frame(root)
    frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    Button(frame, image=cross, command=close).place(relx=0.982, rely=0, relwidth=0.02, relheight=0.03)
    Button(frame, image=mini, command=minimize).place(relx=0.962, rely=0, relwidth=0.02, relheight=0.03)
    Label(frame, text="Enter the Name :", fg="#738f93", font='Helvetica 22 bold') \
        .place(relx=0.004, rely=0.0249, relwidth=0.2, relheight=0.1)
    entry = Entry(frame)
    entry.place(relx=0.24, rely=0.0429, relwidth=0.3, relheight=0.04)
    Button(frame, text="Record Face", bg="#013A55", font="Helvetica 18 bold",
           command=lambda: Face_Record.facerec(entry.get())).place(relx=0.244, rely=0.2, relwidth=0.2, relheight=0.12)

    Button(frame, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
           command=lambda: Menu([frame])).place(relx=0.29, rely=0.933, relwidth=0.24, relheight=0.05)


def facerecog(old_frame):
    old_frame.destroy()
    frame = Frame(root)
    frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    Button(frame, image=cross, command=close).place(relx=0.982, rely=0, relwidth=0.02, relheight=0.03)
    Button(frame, image=mini, command=minimize).place(relx=0.962, rely=0, relwidth=0.02, relheight=0.03)
    new_user = Button(frame, text="Face Record (New User)", bg="#013A55", font="Helvetica 18 bold",
                      command=lambda: newFaceRecord(frame))
    new_user.place(relx=0.044, rely=0.026, relwidth=0.4, relheight=0.2)

    new_user = Button(frame, text="Face Recognition (Old User)", bg="#013A55", font="Helvetica 18 bold",
                      command=Face_Recognition.face_rec)

    new_user.place(relx=0.044, rely=0.48, relwidth=0.4, relheight=0.2)
    main_menu = Button(frame, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                  command=lambda: Menu([frame]))
    main_menu.place(relx=0.29, rely=0.933, relwidth=0.24, relheight=0.05)


# def background_image(path,frame):
#     image = Image.open(path)
#     global copy_of_image
#     copy_of_image= image.copy()
#     photo = ImageTk.PhotoImage(image)
#     label = ttk.Label(frame, image=photo)
#     label.bind('<Configure>', resize_image)
#     label.pack(fill=BOTH, expand=YES)
#     image = Image.open('back.png')


def Menu(old_frames):
    for old_frame in old_frames:
        if old_frame is None:
            continue
        old_frame.destroy()
    frame = Frame(root)
    frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    Button(frame, image=cross, command=close).place(relx=0.982, rely=0, relwidth=0.02, relheight=0.03)
    Button(frame, image=mini, command=minimize).place(relx=0.962, rely=0, relwidth=0.02, relheight=0.03)
    # background_image('Files\\Gui_images\\back.png',root)
    # photo = PhotoImage(file=path)
    #
    # Button(frame, image=photo, command=lambda: Menu([frame])).pack()
    # Label(frame, text="Click on Image to close").pack()
    Titanic = Button(frame, text="Titanic Survivor Problem", bg="#013A55", font="Helvetica 18 bold",
                     command=lambda: Titanic_survivor(frame))
    Titanic.place(relx=0.044, rely=0.015, relwidth=0.4, relheight=0.20)
    Face_recog = Button(frame, text="Face Recognition", bg="#013A55", font="Helvetica 18 bold",
                        command=lambda: facerecog(frame))
    Face_recog.place(relx=0.5, rely=0.015, relwidth=0.4, relheight=0.20)
    Emoji_predict = Button(frame, text="Emoji Predictor", bg="#013A55", font="Helvetica 18 bold",
                           command=lambda: emojipredict(frame))
    Emoji_predict.place(relx=0.044, rely=0.23, relwidth=0.4, relheight=0.20)
    flower_predict = Button(frame, text="Flower Recognizer", bg="#013A55", font="Helvetica 18 bold",
                            command=lambda: flower_recognition(frame))
    flower_predict.place(relx=0.5, rely=0.23, relwidth=0.4, relheight=0.20)
    odd = Button(frame, text="Odd one out", bg="#013A55", font="Helvetica 18 bold",
                 command=lambda: odd_out(frame))
    odd.place(relx=0.5, rely=0.46, relwidth=0.4, relheight=0.2)
    word_analogy = Button(frame, text="Word Analogy", bg="#013A55", font="Helvetica 18 bold",
                          command=lambda: word_analog(frame))
    word_analogy.place(relx=0.044, rely=0.46, relwidth=0.4, relheight=0.2)
    spam_ham = Button(frame, text="Spam or Ham", bg="#013A55", font="Helvetica 18 bold",
                      command=lambda: spam_or_ham(frame))
    spam_ham.place(relx=0.5, rely=0.69, relwidth=0.4, relheight=0.2)
    gender_recog = Button(frame, text="Gender Reccognition", bg="#013A55", font="Helvetica 18 bold",
                          command=lambda: gender(frame))
    gender_recog.place(relx=0.044, rely=0.69, relwidth=0.4, relheight=0.2)
    main_menu = Button(frame, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                  command=lambda: Menu([frame]))
    main_menu.place(relx=0.29, rely=0.933, relwidth=0.24, relheight=0.05)


# def resize_image(event):
#     new_width = event.width
#     new_height = event.height
#     image = copy_of_image.resize((new_width, new_height))
#     photo = ImageTk.PhotoImage(image)
#     label.config(image=photo)
#     label.image = photo  # avoid garbage collection


def about():
    text.destroy()
    frame1.destroy()
    frame = Frame(root)
    frame.place(relx=0.026, rely=0.026, relwidth=0.5, relheight=0.9)

    about_us = Label(frame, text=About, bg="#013A55", fg="#00CDF8", font="Helvetica 18 bold")
    about_us.place(relx=0.026, rely=0.026, relwidth=0.95, relheight=0.95)
    label1 = Label(frame, text="ABOUT APPLICATION", font="Helvetica 18 bold", bg='#013A55', fg="white")
    label1.place(relx=0.14, rely=0.1, relwidth=0.7, relheight=0.05)

    main_menu = Button(root, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                       command=lambda: Menu([frame]))
    main_menu.place(relx=0.29, rely=0.933, relwidth=0.24, relheight=0.05)


class A(threading.Thread):
    def run(self):
        glove = open("Files/Emoji_Predictor/glove.6B.50d.txt", encoding='utf-8')
        for line in glove:
            value = line.split()
            word = value[0]
            coefficient = np.asarray(value[1:], dtype=float)
            glove_dictionary[word] = coefficient
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast("Emoji Module is Ready to Run")
        # word_vector = KeyedVectors.load_word2vec_format(
        #     "odd_one_out/Files/GoogleNews-vectors-negative300.bin", binary=True)
        # toaster.show_toast("Odd One Out and Word Analogy is ready to run")


def close():
    sys.exit(0)


def minimize():
    root.iconify()


if __name__ == "__main__":
    obj = A()
    obj.start()
    glove_dictionary = {}
    word_vector = {}

    # word_vector = KeyedVectors.load_word2vec_format(
    #     " G:\Project\Fun With ML\Code\odd_one_out\Files\GoogleNews-vectors-negative300.bin", binary=True)

    root = Tk()
    root.title("Fun With ML & DL")
    root.attributes("-fullscreen", True)

    root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    # root.iconify();

    About = "\n\n\n\n\nWe aim to provide a Learning \nPlatform for the Beginners in\n " \
            "Machine Learning and Deep \nLearning with the help of various \nModels " \
            "of ML we will run some\n code and try to explain how \nthe code works in the" \
            "\nbackend along with some theory\n and each line of code will\n be explained" \
            "\n\n\n\n\nDevloped By:\n   Manish Sharma\n   Chetan Sharma"

    # image = Image.open('cross.png')
    # copy_of_image = image.copy()
    # photo = ImageTk.PhotoImage(image)
    # label = ttk.Label(root, image=photo)
    # label.bind('<Configure>', resize_image)
    # label.pack(fill=BOTH, expand=YES)
    cross = PhotoImage(file='Files\\Gui_images\\cross.png')
    mini = PhotoImage(file='Files\\Gui_images\\minimise.png')

    frame = Frame(root)
    frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    Button(frame, image=cross, command=close).place(relx=0.982, rely=0, relwidth=0.02, relheight=0.03)
    Button(frame, image=mini, command=minimize).place(relx=0.962, rely=0, relwidth=0.02, relheight=0.03)
    text = Label(root, text="Fun With\nMachine Learning\n& Deep Learning", fg="#738f93", font='Helvetica 44 bold',
                 anchor=W)
    text.place(relx=0.1, rely=0.2, relwidth=0.9, relheight=0.53)
    frame1 = Frame(root)
    frame1.place(relx=0.1, rely=0.7, relwidth=0.5, relheight=0.2)

    Continue = Button(frame1, text="Hello, Machine", command=lambda: Menu([text, frame1, frame]), fg="white",
                      bg="#013A55")
    Continue.place(relx=0.0, rely=0.0, relwidth=0.15, relheight=0.3)
    info = Button(frame1, text="About Us", command=about, fg="white", bg="#013A55").place(relx=0.59, rely=0.0,
                                                                                          relwidth=0.15, relheight=0.3)

    root.mainloop()
