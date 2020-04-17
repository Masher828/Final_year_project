import threading
import tkinter.filedialog
from tkinter import *
from win10toast import ToastNotifier
import numpy as np
from PIL import ImageTk, Image
from win32api import GetSystemMetrics


import emojii, Flowers


def second_menu(old_frame, module, func):
    old_frame.destroy()
    second_menu_frame = Frame(root, bg="white").place(relx=0, rely=0, relwidth=1, relheight=1)
    Label(second_menu_frame, text=module, bg="#3838df", fg="white", font="Helvetica 30 bold ", anchor=W).place(relx=0,
                                                                                                               rely=0,
                                                                                                               relwidth=1,
                                                                                                               relheight=0.1)
    Button(second_menu_frame, text="Menu", bd=0, bg="#3838df", fg="white", font="Helvetica 18 bold",
           command=lambda: main_menu([frame])).place(relx=0.8, rely=0.03, relwidth=0.05, relheight=0.05)
    toolbar_and_menu(second_menu_frame, module)
    img = get_image(module + " big")

    Label(second_menu_frame, text=module, fg="black", bg="white", font="Helvetica 38 bold").place(relx=0.001, rely=0.17,
                                                                                                  relwidth=0.4,
                                                                                                  relheight=0.1)
    Label(second_menu_frame, text="will add through .txt file same sa that in menu", bg="white", font="Helvetica 15",
          anchor=W).place(relx=0.013, rely=0.27, relwidth=0.5, relheight=0.2)
    Label(second_menu_frame, image=img, bg="white").place(relx=0.8, rely=0.4, anchor=CENTER)

    ImageButton(second_menu_frame, second_image=get_image("welcome_start_red"), image=get_image("welcome_start_red"),
                bg="white", bd=0,
                command=lambda: func(second_menu_frame, 1)).place(relx=0.013, rely=0.5,
                                                                  relwidth=0.2, relheight=0.07)
    ImageButton(second_menu_frame, second_image=get_image("about_algo"), image=get_image("about_algo"), bg="white",
                bd=0,
                command=lambda: func(second_menu_frame, 1)).place(relx=0.013, rely=0.8,
                                                                  relwidth=0.08, relheight=0.07)
    ImageButton(second_menu_frame, second_image=get_image("Working"), image=get_image("Working"), bg="white",
                bd=0,
                command=lambda: func(second_menu_frame, 1)).place(relx=0.1, rely=0.8,
                                                                  relwidth=0.09, relheight=0.07)
    root.mainloop()


def about_algo(module):
    pass


def module_working(module):
    pass


def word_analog(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Word Analogy", word_analog)
    word_analogy_frame = Frame(root)
    word_analogy_frame.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)
    toolbar_and_menu(word_analogy_frame, "Word Analogy")
    word_a = Label(word_analogy_frame, text="Enter word A", fg="white", bg="#013A55",
                   font='Helvetica 10 bold')
    word_a.place(relx=0.1, rely=0.1, relwidth=0.27, relheight=0.09)
    a = Entry(word_analogy_frame)
    a.place(relx=0.4, rely=0.1, relwidth=0.27, relheight=0.09)
    word_b = Label(word_analogy_frame, text="Enter word B", fg="white", bg="#013A55",
                   font='Helvetica 10 bold')
    word_b.place(relx=0.1, rely=0.3, relwidth=0.27, relheight=0.09)
    b = Entry(word_analogy_frame)
    b.place(relx=0.4, rely=0.3, relwidth=0.27, relheight=0.1)
    word_c = Label(word_analogy_frame, text="Enter word C", fg="white", bg="#013A55",
                   font='Helvetica 10 bold')
    word_c.place(relx=0.1, rely=0.6, relwidth=0.37, relheight=0.09)
    c = Entry(word_analogy_frame)
    c.place(relx=0.4, rely=0.6, relwidth=0.2, relheight=0.1)
    submit = Button(root, text="Predict Word", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                    command=lambda: predict_word(word_analogy_frame, [a.get(), b.get(), c.get()]))
    submit.place(relx=0.7, rely=0.7, relwidth=0.2, relheight=0.1)


def predict_word(word_analogy_frame, words):
    odd = word_analogy.word_analogyy(glove_dictionary, words)
    Button(word_analogy_frame, text=odd, command=lambda: main_menu([frame])).place(relx=0.7, rely=0.5, relwidth=0.2,
                                                                                   relheight=0.2)


def odd_out(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Odd One Out", odd_out)
    odd_out_frame = Frame(root)
    odd_out_frame.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)
    toolbar_and_menu(odd_out_frame, "Odd One Out")
    label = Label(odd_out_frame, text="Enter the words seprated by comma (,) ", fg="white", bg="#013A55",
                  font='Helvetica 10 bold')
    label.place(relx=0.1, rely=0.1, relwidth=0.37, relheight=0.09)
    sent = Entry(odd_out_frame)
    sent.place(relx=0.1, rely=0.3, relwidth=0.2, relheight=0.1)
    submit = Button(odd_out_frame, text="Find odd word ", command=lambda: odd_name(sent.get(), odd_out_frame))
    submit.place(relx=0.15, rely=0.43, relwidth=0.1, relheight=0.1)
    main_menu = Button(odd_out_frame, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                       command=lambda: main_menu([odd_out_frame]))
    main_menu.place(relx=0.29, rely=0.933, relwidth=0.24, relheight=0.05)


def odd_name(sent, odd_out_frame):
    odd = odd_one_out.externalodd_out(glove_dictionary, sent)
    Button(odd_out_frame, text=odd, command=lambda: main_menu([frame])).place(relx=0.5, rely=0.5, relwidth=0.2,
                                                                              relheight=0.2)


def flower_recognition(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Flower Recognition", flower_recognition)
    flower_recognition_frame = Frame(root)
    flower_recognition_frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    toolbar_and_menu(flower_recognition_frame, "Flower Recognition")
    filename_path = tkinter.filedialog.askopenfilename()
    label = Label(flower_recognition_frame, text=filename_path, fg="white", bg="#013A55", font='Helvetica 10 bold')
    label.place(relx=0.06, rely=0.06, relwidth=0.37, relheight=0.09)

    submit = Button(flower_recognition_frame, text="Recognize", bg="#013A55", font="Helvetica 18 bold",
                    command=lambda: recog_flower_name(filename_path, flower_recognition_frame))
    submit.place(relx=0.31, rely=0.19, relwidth=0.1, relheight=0.1)


def recog_flower_name(filename_path, flower_recog_frame):
    path = 'Files/Image_Classification/Output/'
    flower_name = Flowers.flower_recog(filename_path)
    img = PhotoImage(file=path + flower_name + ".png")
    data = open(path + flower_name + ".txt", encoding="utf8")
    contents = data.read()
    flower_recog_frame.destroy()
    Flower_details_frame = Frame(root)
    Flower_details_frame.place(relx=0.02, rely=0.02, relwidth=1, relheight=1)
    toolbar_and_menu(Flower_details_frame, "Flower Recognition")
    Label(Flower_details_frame, text=contents, fg="black", bg="#013A55", font='Helvetica 10').place(relx=0.0, rely=0.3)
    Button(Flower_details_frame, image=img, command=lambda: main_menu([Flower_details_frame])).place(relx=0.35,
                                                                                                     rely=0.0,
                                                                                                     relheight=0.3)
    root.mainloop()


def emojipredict(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Emoji Predictor", emojipredict)
    emoji_predict_frame = Frame(root, bg="white")
    emoji_predict_frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    Label(emoji_predict_frame, bg="#3838df", fg="white", font="Helvetica 30 bold ", anchor=W).place(relx=0,
                                                                                                    rely=0,
                                                                                                    relwidth=1,
                                                                                                    relheight=0.1)
    Label(emoji_predict_frame, text="Emoji Predictor", bg="#3838df", fg="white", font="Helvetica 30 bold ",
          anchor=W).place(relx=0,
                          rely=0,
                          relwidth=1,
                          relheight=0.1)
    toolbar_and_menu(emoji_predict_frame, "Emoji Predictor")
    label = Label(emoji_predict_frame, text="Enter the text here ", fg="#738f93", font='Helvetica 22 bold')
    label.place(relx=0.01, rely=0.1, relwidth=0.2, relheight=0.1)
    sent = Entry(emoji_predict_frame)
    sent.place(relx=0.25, rely=0.1, relwidth=0.2, relheight=0.1)
    submit = Button(emoji_predict_frame, text="Get Emoji", bg="#013A55", font="Helvetica 18 bold",
                    command=lambda: emojipredict_image(glove_dictionary, sent.get(), emoji_predict_frame))
    submit.place(relx=0.30, rely=0.23, relwidth=0.1, relheight=0.1)
    Button(emoji_predict_frame, text="Menu", bd=0, bg="#3838df", fg="white", font="Helvetica 18 bold",
           command=lambda: main_menu([frame])).place(relx=0.8, rely=0.03, relwidth=0.05, relheight=0.05)


def emojipredict_image(glove_dicitionary, sent, emoji_predict_frame):
    path = emojii.pred(glove_dicitionary, sent)
    photo = PhotoImage(file=path)
    Button(emoji_predict_frame, image=photo, command=lambda: main_menu([emoji_predict_frame])).place(relx=0.1, rely=0.4)
    Label(emoji_predict_frame, text="Click on Image to close").place(relx=0.034, rely=0.87, relwidth=0.3, relheight=0.1)
    root.mainloop()


def newFaceRecord(old_frame):
    old_frame.destroy()
    face_record_frame = Frame(root)
    face_record_frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    toolbar_and_menu(face_record_frame, "Face Classification")
    Label(face_record_frame, text="Enter the Name :", fg="#738f93", font='Helvetica 22 bold') \
        .place(relx=0.004, rely=0.0249, relwidth=0.2, relheight=0.1)
    entry = Entry(face_record_frame)
    entry.place(relx=0.24, rely=0.0429, relwidth=0.3, relheight=0.04)
    Button(face_record_frame, text="Record Face", bg="#013A55", font="Helvetica 18 bold",
           command=lambda: Face_Record.facerec(entry.get())).place(relx=0.244, rely=0.2, relwidth=0.2, relheight=0.12)


def facerecog(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Face Classification", facerecog)
    face_recog_frame = Frame(root, bg="white")
    face_recog_frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    toolbar_and_menu(face_recog_frame, "Face Classification")
    new_user = ImageButton(face_recog_frame, image=get_image("facerecordnew"), second_image=get_image("facerecordnew"),
                           bg="white", font="Helvetica 18 bold", bd=0,
                           command=lambda: newFaceRecord(face_recog_frame))
    new_user.place(relx=0.144, rely=0.17, relwidth=0.35, relheight=0.2)

    new_user = ImageButton(face_recog_frame, image=get_image("facerecordold"), second_image=get_image("facerecordold"),
                           bg="white", font="Helvetica 18 bold", bd=0,
                           command=Face_Recognition.face_rec)

    new_user.place(relx=0.144, rely=0.41, relwidth=0.35, relheight=0.2)



def toolbar_and_menu(module_frame, module):
    Label(module_frame, bg="#3838df", fg="white", font="Helvetica 30 bold ", anchor=W).place(relx=0,
                                                                                             rely=0,
                                                                                             relwidth=1,
                                                                                             relheight=0.1)
    Label(module_frame, text=module, bg="#3838df", fg="white", font="Helvetica 30 bold ",
          anchor=W).place(relx=0,
                          rely=0,
                          relwidth=1,
                          relheight=0.1)
    cross = get_image("close_red_blue")
    cross2 = get_image("close_red_blue_cross")
    mini = get_image("mini_yellow_blue")
    mini2 = get_image("mini_yellow_blue_min")
    bg = "#3838df"

    ImageButton(module_frame, image=cross, second_image=cross2, command=close, border="0", bg=bg,
                bd="0").place(relx=0.982,
                              rely=0,
                              relwidth=0.02,
                              relheight=0.03)
    ImageButton(module_frame, image=mini, second_image=mini2, command=minimize, border="0", bg=bg,
                bd="0").place(
        relx=0.962, rely=0, relwidth=0.018, relheight=0.03)
    Button(module_frame, text="Menu", bd=0, bg="#3838df", fg="white", font="Helvetica 18 bold",
           command=lambda: main_menu([frame])).place(relx=0.8, rely=0.03, relwidth=0.05, relheight=0.05)


def get_image(module):
    im = Image.open("Files/Gui_images/" + module + ".png")
    ph = ImageTk.PhotoImage(im)
    return ph


def spam_or_ham(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Spam or Ham", spam_or_ham)
    spam_or_ham_frame = Frame(root, bg="white")
    spam_or_ham_frame.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)
    toolbar_and_menu(spam_or_ham_frame, "Spam or Ham")

    comm = Entry(spam_or_ham_frame)
    comm.place(relx=0.4, rely=0.4, relwidth=0.27, relheight=0.09)
    submit = Button(root, text="Predict Word", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                    command=lambda: predict_spam(spam_or_ham_frame, [comm.get]))
    submit.place(relx=0.4, rely=0.55, relwidth=0.2, relheight=0.1)

    Button(spam_or_ham_frame, text="Menu", bd=0, bg="#3838df", fg="white", font="Helvetica 18 bold",
           command=lambda: main_menu([frame])).place(relx=0.8, rely=0.01, relwidth=0.15, relheight=0.08)

    img = get_image("submit")
    Label(spam_or_ham_frame, image=img, bg="white", bd=0).place(relx=0.1, rely=0.4, relwidth=0.27, relheight=0.15)
    root.mainloop()


# def predict_spam(spam_or_ham_frame):

def character_recognition(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Character Recognition", character_recognition)


def sentiment_analysis(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Sentiment Analysis", sentiment_analysis)
    sentiment_analysis_frame = Frame(root, bg="white")
    sentiment_analysis_frame.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)
    toolbar_and_menu(sentiment_analysis_frame, "Sentiment Analysis")

    comm = Entry(sentiment_analysis_frame)
    comm.place(relx=0.4, rely=0.4, relwidth=0.27, relheight=0.09)
    submit = Button(root, text="Predict Word", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                    command=lambda: predict_sentiment(sentiment_analysis_frame, [comm.get]))
    submit.place(relx=0.4, rely=0.55, relwidth=0.15, relheight=0.1)
    Button(sentiment_analysis_frame, text="Menu", bd=0, bg="#3838df", fg="white", font="Helvetica 18 bold",
           command=lambda: main_menu([frame])).place(relx=0.8, rely=0.01, relwidth=0.15, relheight=0.08)

    img = get_image("submit")
    Label(sentiment_analysis_frame, image=img, bg="white", bd=0).place(relx=0.1, rely=0.4, relwidth=0.27,
                                                                       relheight=0.15)
    root.mainloop()


# def predict_sentiment(sentiment_analysis_frame):


def titanic_survivor(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Titanic Survivor", titanic_survivor)


def main_menu(old_frames):
    for old_frame in old_frames:
        if old_frame is None:
            continue
        old_frame.destroy()
    wid = GetSystemMetrics(0)
    hig = GetSystemMetrics(1)
    cross = get_image("close_red_blue")
    cross2 = get_image("close_red_blue_cross")
    mini = get_image("mini_yellow_blue")
    mini2 = get_image("mini_yellow_blue_min")
    full_frame = Frame(root, bg="#3838df").place(relx=0, rely=0, relwidth=1, relheight=1)
    ImageButton(full_frame, image=cross, second_image=cross2, command=close, border="0", bg="#3838df",
                bd=0).place(relx=0.982,
                            rely=0,
                            relwidth=0.02,
                            relheight=0.03)
    ImageButton(full_frame, image=mini, second_image=mini2, command=minimize, border="0", bg="#3838df",
                bd=0).place(
        relx=0.962, rely=0, relwidth=0.018, relheight=0.03)
    Label(full_frame, text="Modules", bg="#3838df", fg="white", font="Helvetica 40 bold ").grid(row=0, column=0)
    menu_temp_frame = Frame(full_frame, bg="white", relief=GROOVE, bd=1)
    menu_temp_frame.place(relx=0.0, rely=0.1, relwidth=1, relheight=1)
    canvas = Canvas(menu_temp_frame, bg="white")
    Menu_frame = Frame(canvas, bg="white")
    myscrollbar = Scrollbar(menu_temp_frame, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=myscrollbar.set)

    def myfunction(event):
        canvas.configure(scrollregion=canvas.bbox("all"), width=wid, height=hig)

    myscrollbar.pack(side="right", fill="y")
    canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
    canvas.create_window((0, 0), window=Menu_frame, anchor='nw')
    Menu_frame.bind("<Configure>", myfunction)

    image = get_image("Titanic Survivor")
    description_img = get_image("titanic_survivor_about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: titanic_survivor(frame)).grid(row=1, column=0)

    image = get_image("Face Classification")
    description_img = get_image("face_recog_about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: facerecog(frame)).grid(row=1, column=1)

    image = get_image("Emoji Predictor")
    description_img = get_image("emoji_predict_about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: emojipredict(Menu_frame)).grid(row=1, column=2)

    image = get_image("Flower Recognition")
    description_img = get_image("flower_recog_about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: flower_recognition(Menu_frame)).grid(row=1, column=3)

    image = get_image("Odd One Out")
    description_img = get_image("odd one about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: odd_out(Menu_frame)).grid(row=2, column=0)

    image = get_image("Word Analogy")
    description_img = get_image("word analogy about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: word_analog(Menu_frame)).grid(row=2, column=1)

    image = get_image("Spam or Ham")
    description_img = get_image("spam or ham about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: spam_or_ham(Menu_frame)).grid(row=2, column=2)

    image = get_image("Gender Prediction")
    description_img = get_image("gender predict about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: gender(Menu_frame)).grid(row=2, column=3)

    image = get_image("Character Recognition")
    description_img = get_image("character recog about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: character_recognition(Menu_frame)).grid(row=3, column=0)

    image = get_image("Sentiment Analysis")
    description_img = get_image("senti anal about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: sentiment_analysis(Menu_frame)).grid(row=3, column=1)

    Button(Menu_frame, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
           command=lambda: main_menu([frame])).grid(row=6, column=1)


class ImageButton(Button):
    def __init__(self, master, second_image, image, **kw):
        Button.__init__(self, master=master, **kw)
        self['image'] = image
        self.default_image = image
        self.second_image = second_image
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self['image'] = self.second_image

    def on_leave(self, e):
        self['image'] = self.default_image


def about():
    text.destroy()
    about_us_frame = Frame(root)
    about_us_frame.place(relx=0.026, rely=0.026, relwidth=0.5, relheight=0.9)

    about_us = Label(about_us_frame, text=About, bg="#013A55", fg="#00CDF8", font="Helvetica 18 bold")
    about_us.place(relx=0.026, rely=0.026, relwidth=0.95, relheight=0.95)
    label1 = Label(about_us_frame, text="ABOUT APPLICATION", font="Helvetica 18 bold", bg='#013A55', fg="white")
    label1.place(relx=0.14, rely=0.1, relwidth=0.7, relheight=0.05)

    menu = Button(root, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
                  command=lambda: main_menu([about_us_frame]))
    menu.place(relx=0.29, rely=0.933, relwidth=0.24, relheight=0.05)


class A(threading.Thread):
    def run(self):
        global Face_Recognition, Face_Record, odd_one_out, word_analogy
        sys.path.insert(1, "Face_Recog/")
        sys.path.insert(4, "odd_one_out/")
        sys.path.insert(5, "Word_analogy/")

        Face_Recognition = __import__('Face_Recognition', globals())
        Face_Record = __import__('Face_Record', globals())
        odd_one_out = __import__('odd_one_out', globals())
        word_analogy = __import__('word_analogy', globals())
        glove = open("Files/Emoji_Predictor/glove.6B.50d.txt", encoding='utf-8')
        for line in glove:
            value = line.split()
            word = value[0]
            coefficient = np.asarray(value[1:], dtype=float)
            glove_dictionary[word] = coefficient
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

    About = "\n\n\n\n\nWe aim to provide a Learning \nPlatform for the Beginners in\n " \
            "Machine Learning and Deep \nLearning with the help of various \nModels " \
            "of ML we will run some\n code and try to explain how \nthe code works in the" \
            "\nbackend along with some theory\n and each line of code will\n be explained" \
            "\n\n\n\n\nDeveloped By:\n   Manish Sharma\n   Chetan Sharma"

    frame = Frame(root, bg="white")
    frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)

    text = Label(frame, text="FUN WITH             \nMachine Learning\n& Deep Learning  ", fg="#125bda",
                 font='Helvetica 44 bold', bg="white")
    text.place(relx=0.05, rely=0.25, relwidth=0.4, relheight=0.3)

    img = get_image("welcome_start_red")
    start = ImageButton(frame, second_image=img, image=img, command=lambda: main_menu([frame]), border="0", bg="white")
    img = get_image("welcome_about_red")
    start.place(relx=0.05, rely=0.7, relwidth=0.19, relheight=0.1)
    welcome_about = ImageButton(frame, second_image=img, image=img, command=about, border="0", bg="white").place(
        relx=0.77, rely=0.03,
        relwidth=0.19, relheight=0.1)
    img = get_image("welcome_img")
    Label(frame, image=img, bg="white").place(relx=0.5, rely=0.3, relwidth=0.5, relheight=0.7)

    root.mainloop()
