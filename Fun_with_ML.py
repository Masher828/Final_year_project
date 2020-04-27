import threading
import tkinter.filedialog
from tkinter import *
import re
from win10toast import ToastNotifier
from gensim.models import KeyedVectors
from win10toast import ToastNotifier
import numpy as np
from PIL import ImageTk, Image
from win32api import GetSystemMetrics
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

sys.path.insert(2, "Emoji_predictor/")
sys.path.insert(3, "Image_Classification/")
sys.path.insert(9, "Gender_Recog/")

import emojii , Flowers, gender

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
    img = get_gui_image(module + " big")

    ima=ImageLabel(second_menu_frame, image =img,relx=0.7, rely=0.2)
    Label(second_menu_frame, text="will add through .txt file same sa that in menu", bg="white", font="Helvetica 15",
          anchor=W).place(relx=0.013, rely=0.27, relwidth=0.5, relheight=0.2)
    Label(second_menu_frame, text=module, bg="white", font="Helvetica 25 bold").place(x=0.011, rely=0.17)
    contents=""
    label = Label(second_menu_frame, text=contents, font="Helvetica 10", justify="left")
    label.place(relx=0.46, rely=0.2)

    back=ImageButton(second_menu_frame, second_image=get_gui_image("welcome_start_red"),
                image=get_gui_image("welcome_start_red"),
                bg="white", bd=0,command=lambda: func(second_menu_frame, 1)).place(relx=0.013, rely=0.5)

    about_al= ImageButton(second_menu_frame, second_image=get_gui_image("about_algo"), image=get_gui_image("about_algo"),
                bg="white",
                bd=0,
                command=lambda: about_algo(module,label,ima)).place(relx=0.013, rely=0.8)

    module_work=ImageButton(second_menu_frame, second_image=get_gui_image("Working"), image=get_gui_image("Working"), bg="white",
                bd=0,
                command=lambda: module_working(module,label,ima)).place(relx=0.1, rely=0.8)
    doubt_butt = ImageButton(second_menu_frame, second_image=get_gui_image("doubt"), image=get_gui_image("doubt"),
                bg="white",
                bd=0,
                command=lambda: doubt(second_menu_frame,module,label,ima)).place(relx=0.049, rely=0.9)
    root.mainloop()

def doubt(second_menu_frame,module, label, ima):
    ima.Destroy()
    label["text"]=""
    enter_doubt_label = Label(second_menu_frame, text = "Enter Your Doubt",font = "Helvetica 20 bold",bg = "white" )
    enter_doubt_label.place(relx=0.5,rely =0.1)
    doubt_label = Label(second_menu_frame, text = " ",fg = 'red',bg = "white", font = "Helvetica 10")
    doubt_label.place(relx=0.67, rely=0.11)
    enter_doubt_entry = Text(second_menu_frame, font = "Helvetica 20", height = 11, width =45,bg = "white")
    enter_doubt_entry.place(relx=0.5,rely=0.15)
    enter_email_label = Label(second_menu_frame, text="Enter Your Email",bg = "white", font="Helvetica 20 bold")
    enter_email_label.place(relx=0.5, rely=0.6)
    email_label = Label(second_menu_frame, text=" ", fg='red',bg ="white", font="Helvetica 10")
    email_label.place(relx=0.67, rely=0.61)
    enter_email_entry = Text(second_menu_frame, font="Helvetica 20", height =1.5,width = 45,bg = "white")
    enter_email_entry.place(relx=0.5, rely=0.65)
    to_destroy=[enter_doubt_label,enter_email_entry,enter_doubt_entry,enter_email_label]

    submit_doubt = ImageButton(second_menu_frame, second_image=get_gui_image("submit_doubt"), image=get_gui_image("submit_doubt"),
                bg="white",
                bd=0,
                command=lambda: query(enter_doubt_entry.get("1.0","end-1c"),enter_email_entry.get("1.0","end-1c"),module, doubt_label, email_label,to_destroy,submit_doubt,label))
    submit_doubt.place(relx=0.9, rely=0.9)

def query(doubt,email,subject,doubt_label, email_label,to_destroy,butt,label):
    doubt_ = False
    email_ = False
    regex = '^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$'
    if len(doubt) ==0 :
        doubt_label["text"]="Doubt Field Cannot be Empty"
    else:
        doubt_label["text"]=""
        doubt_ =True
    if len(email) ==0 :
        email_label["text"]="Email cannot be empty, You won't be able to get response of doubt"
    elif not re.search(regex, email) :
        email_label["text"]="Invalid Email"
    else:
        email_label["text"]=""
        email_=True

    if email_ and doubt_ :
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login('yearfinal835@gmail.com', 'MachineLearning')
        msg = MIMEMultipart()
        msg['From'] = 'yearfinal835@gmail.com'
        msg['To'] = 'yearfinal835@gmail.com'
        msg['Subject'] = subject
        msg.attach(MIMEText(doubt+"\n Email id of Person is " + email, 'plain'))
        s.send_message(msg)
        del msg
        s.quit()

        doubt_label.destroy()
        email_label.destroy()
        butt.destroy()
        label["text"]="Doubt is Submitted"
        for label in to_destroy :
            label.destroy()



def about_algo(module,label,ima):
    path = "Files/"
    data = open(path + "_".join(module.split(" ")) + "/about.txt", encoding="utf8")
    contents = data.read()

    if not old is None:
        old.destroy()
    if frame is None:
        about_algo_frame = Frame(root)
        about_algo_frame.place(relx = 0.5, rely = 0.1,relwidth =0.5, relheight = 0.8)
    else:
        about_algo_frame=frame
    label = Label(about_algo_frame,text = contents , font = "Helvetica 10")
    label.place(relx=0, rely=0)
    module_work["command"]= module_working(module,module_work,about_al,label,about_algo_frame)
    # Flower_details_frame.place(relx=0.02, rely=0.02, relwidth=1, relheight=1)
    root.mainloop()

    label["text"]=contents
    ima.Destroy()




def module_working( module,label,ima):
    path = "Files/"
    data = open(path + "_".join(module.split(" ")) + "/steps.txt", encoding="utf8")
    contents = data.read()
    label["text"]=contents
    ima.Destroy()
    


def word_analog(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Word Analogy", word_analog)
    word_analogy_frame = Frame(root, bg="white")
    word_analogy_frame.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)
    toolbar_and_menu(word_analogy_frame, "Word Analogy")
    ImageLabel(word_analogy_frame, image=get_gui_image("worda"), relx=0.03, rely=0.2)
    a = Entry(word_analogy_frame, font="Helvetica 18 bold")
    a.place(relx=0.3, rely=0.21, relwidth=0.17, relheight=0.09)
    ImageLabel(word_analogy_frame, image=get_gui_image("wordb"), relx=0.5, rely=0.2)
    b = Entry(word_analogy_frame, font="Helvetica 18 bold")
    b.place(relx=0.77, rely=0.21, relwidth=0.17, relheight=0.1)
    ImageLabel(word_analogy_frame, image=get_gui_image("wordc"), relx=0.03, rely=0.4)
    c = Entry(word_analogy_frame, font="Helvetica 18 bold")
    c.place(relx=0.3, rely=0.41, relwidth=0.17, relheight=0.09)
    ImageButton(root, image=get_gui_image("predict_word"), second_image=get_gui_image("predict_word"), bg="white",
                 bd=0,
                command=lambda: predict_word(word_analogy_frame, [a.get(), b.get(), c.get()])).place(relx=0.03,
                                                                                                     rely=0.61)


def predict_word(word_analogy_frame, words):
    try :

        print(type(word_vector))
        odd = word_analogy.word_analogyy(word_vector, words)
        print("hi")
    except :

        print(type(glove_dictionary))
        odd = word_analogy.word_analogyy(glove_dictionary, words)
        print("tata")
    ImageLabel(word_analogy_frame, image=get_gui_image("wordd"), relx=0.5, rely=0.41, relwidth=0.26, relheight=0.15)
    Label(word_analogy_frame, text = odd, font="Helvetica 18 bold").place(relx=0.77, rely=0.2, relwidth=0.26, relheight=0.15)


def odd_out(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Odd One Out", odd_out)
    odd_out_frame = Frame(root, bg="white")
    odd_out_frame.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)
    toolbar_and_menu(odd_out_frame, "Odd One Out")
    ImageLabel(odd_out_frame, image=get_gui_image("oddd"), relx=0.03, rely=0.2, relwidth=0.26, relheight=0.15)
    sent = Entry(odd_out_frame, font="Helvetica 30 bold ")
    sent.place(relx=0.31, rely=0.21, relwidth=0.35, relheight=0.1)
    ImageButton(odd_out_frame, image=get_gui_image("odd_word"), second_image=get_gui_image("odd_word"),
                bg="white", font="Helvetica 18 bold", bd=0,
                command=lambda: odd_name(sent.get(), odd_out_frame)).place(relx=0.31, rely=0.35)


def odd_name(sent, odd_out_frame):
    odd = odd_one_out.externalodd_out(glove_dictionary, sent)
    Button(odd_out_frame, text=odd, command=lambda: main_menu([frame])).place(relx=0.5, rely=0.5, relwidth=0.2,
                                                                              relheight=0.2)


global filename_path


def flower_recognition(old_frame, check_call=None):
    if check_call is None:
        print("hello")
        second_menu(old_frame, "Flower Recognition", flower_recognition)
    flower_recognition_frame = Frame(root, bg="white")
    flower_recognition_frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    toolbar_and_menu(flower_recognition_frame, "Flower Recognition")

    def choose(label):
        label["text"] = tkinter.filedialog.askopenfilename()
        # ImageLabel(flower_recognition_frame, image=get_image(label["text"]), relx=0.68, rely=0.24, relwidth=0.3, relheight=0.3)

    label = Label(flower_recognition_frame, text="Path of the File", font='Helvetica 10')
    label.place(relx=0.03, rely=0.2, relwidth=0.48, relheight=0.15)
    chose = ImageButton(flower_recognition_frame, image=get_gui_image("choose_flower"),
                        second_image=get_gui_image("choose_flower"), bg="white", font="Helvetica 18 bold", bd=0,
                        command=lambda: choose(label))
    chose.place(relx=0.03, rely=0.35)

    submit = ImageButton(flower_recognition_frame, image=get_gui_image("recognize_flower"),
                         second_image=get_gui_image("recognize_flower"), bd=0, bg="white", font="Helvetica 18 bold",
                         command=lambda: recog_flower_name(label["text"], flower_recognition_frame))
    submit.place(relx=0.31, rely=0.35)


def recog_flower_name(filename_path, flower_recog_frame):
    path = 'Files/Image_Classification/Output/'
    flower_name = Flowers.flower_recog(filename_path)
    img = PhotoImage(file=path + flower_name + ".png")
    # data = open(path + flower_name + ".txt", encoding="utf8")
    # contents = data.read()
    # flower_recog_frame.destroy()
    # Flower_details_frame = Frame(root)
    # Flower_details_frame.place(relx=0.02, rely=0.02, relwidth=1, relheight=1)
    toolbar_and_menu(flower_recog_frame, "Flower Recognition")
    # ImageLabel(flower_recog_frame, text=contents, fg="black", bg="#013A55", font='Helvetica 10').place(relx=0.0, rely=0.3)
    ImageLabel(flower_recog_frame, image=img, relx=0.68, rely=0.50)
    Label(flower_recog_frame, text=flower_name, bg="white").place(relx=0.68, rely=0.80, relwidth=0.1, relheight=0.1)


def emojipredict(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Emoji Predictor", emojipredict)
    emoji_predict_frame = Frame(root, bg="white")
    emoji_predict_frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    toolbar_and_menu(emoji_predict_frame, "Emoji Predictor")
    ImageLabel(emoji_predict_frame, image=get_gui_image("emoji"), relx=0.03, rely=0.2)
    sent = Entry(emoji_predict_frame, font="Helvetica 30 bold ")
    sent.place(relx=0.31, rely=0.21, relwidth=0.35, relheight=0.1)
    submit = ImageButton(emoji_predict_frame, image=get_gui_image("get_emoji"), second_image=get_gui_image("get_emoji"),
                         bg="white", font="Helvetica 18 bold", bd=0,
                         command=lambda: emojipredict_image(glove_dictionary, sent.get(), emoji_predict_frame))
    submit.place(relx=0.31, rely=0.35)


def emojipredict_image(glove_dicitionary, sent, emoji_predict_frame):
    path = emojii.pred(glove_dicitionary, sent)
    ImageLabel(emoji_predict_frame, image=get_image(path), relx=0.75, rely=0.45)


def newFaceRecord(old_frame):
    old_frame.destroy()
    face_record_frame = Frame(root, bg="white")
    face_record_frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    toolbar_and_menu(face_record_frame, "Face Classification")
    ImageLabel(face_record_frame, image=get_gui_image("enter_name"), relx=0.03, rely=0.2, relwidth=0.26, relheight=0.15)

    entry = Entry(face_record_frame, font="Helvetica 30 bold ")
    entry.place(relx=0.31, rely=0.21, relwidth=0.35, relheight=0.1)
    ImageButton(face_record_frame, image=get_gui_image("record_face"), second_image=get_gui_image("record_face"),
                bg="white", font="Helvetica 18 bold", bd=0,
                command=lambda: Face_Record.facerec(entry.get())).place(relx=0.31, rely=0.35)


def facerecog(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Face Classification", facerecog)
    face_recog_frame = Frame(root, bg="white")
    face_recog_frame.place(relx=0.0, rely=0.0, relwidth=1, relheight=1)
    toolbar_and_menu(face_recog_frame, "Face Classification")
    new_user = ImageButton(face_recog_frame, image=get_gui_image("facerecordnew"),
                           second_image=get_gui_image("facerecordnew"),
                           bg="white", font="Helvetica 18 bold", bd=0,
                           command=lambda: newFaceRecord(face_recog_frame))
    new_user.place(relx=0.03, rely=0.2)

    new_user = ImageButton(face_recog_frame, image=get_gui_image("facerecordold"),
                           second_image=get_gui_image("facerecordold"),
                           bg="white", font="Helvetica 18 bold", bd=0,
                           command=Face_Recognition.face_rec)

    new_user.place(relx=0.03, rely=0.45)


def toolbar_and_menu(module_frame, module):
    Label(module_frame, bg="#3838df", fg="white", font="Helvetica 30 bold ", anchor=W).place(relx=0,
                                                                                             rely=0,
                                                                                             relwidth=1,
                                                                                             relheight=0.1)
    Label(module_frame, text=module, bg="#3838df", fg="white", font="Helvetica 30 bold ",
          anchor=W).place(relx=0,
                          rely=0)
    cross = get_gui_image("close_red_blue")
    cross2 = get_gui_image("close_red_blue_cross")
    mini = get_gui_image("mini_yellow_blue")
    mini2 = get_gui_image("mini_yellow_blue_min")
    bg = "#3838df"

    ImageButton(module_frame, image=cross, second_image=cross2, command=close, border="0", bg=bg,
                bd="0").place(relx=0.982,
                              rely=0)
    ImageButton(module_frame, image=mini, second_image=mini2, command=minimize, border="0", bg=bg,
                bd="0").place(
        relx=0.962, rely=0)
    Button(module_frame, text="Menu", bd=0, bg="#3838df", fg="white", font="Helvetica 18 bold", cursor = "hand2",
           command=lambda: main_menu([frame])).place(relx=0.8, rely=0.03)


def get_gui_image(module):
    im = Image.open("Gui_images/" + module + ".png")
    ph = ImageTk.PhotoImage(im)
    return ph


def get_image(module):
    im = Image.open(module)
    ph = ImageTk.PhotoImage(im)
    return ph


def spam_or_ham(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Spam or Ham", spam_or_ham)
    spam_or_ham_frame = Frame(root, bg="white")
    spam_or_ham_frame.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)
    toolbar_and_menu(spam_or_ham_frame, "Spam or Ham")
    ImageLabel(spam_or_ham_frame, image=get_gui_image("submit"), relx=0.03, rely=0.2)

    sent = Entry(spam_or_ham_frame,font="Helvetica 30 bold ")
    sent.place(relx=0.31, rely=0.21, relwidth=0.35, relheight=0.1)
    ImageButton(spam_or_ham_frame, image=get_gui_image("sentiment12"), second_image=get_gui_image("sentiment12"),
                bg="white", font="Helvetica 18 bold", bd=0,
                command=lambda: predict_spam(sent.get(), spam_or_ham_frame)).place(relx=0.31, rely=0.35)


def predict_spam(sent,spam_or_ham_frame):
    spamm=Spam.analyzer(sent)
    Label(spam_or_ham_frame, text=spamm, bg="white",font="Helvetica 30 bold ").place(relx=0.4, rely=0.7, relwidth=0.2, relheight=0.1)


def character_recognition(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Character Recognition", character_recognition)


def sentiment_analysis(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Sentiment Analysis", sentiment_analysis)
    sentiment_analysis_frame = Frame(root, bg="white")
    sentiment_analysis_frame.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)
    toolbar_and_menu(sentiment_analysis_frame, "Sentiment Analysis")

    ImageLabel(sentiment_analysis_frame, image=get_gui_image("submit"), relx=0.03, rely=0.2)

    sent = Entry(sentiment_analysis_frame,font="Helvetica 30 bold ")
    sent.place(relx=0.31, rely=0.21, relwidth=0.35, relheight=0.1)
    ImageButton(sentiment_analysis_frame, image=get_gui_image("sentiment12"), second_image=get_gui_image("sentiment12"),
                bg="white", font="Helvetica 18 bold", bd=0,
                command=lambda: predict_sentiment(sent.get(), sentiment_analysis_frame)).place(relx=0.31, rely=0.35)


def predict_sentiment(sent, sentiment_analysis_frame):
    mood = sentiment.mood_analyzer(sent)
    Label(sentiment_analysis_frame, text=mood, bg="white",font="Helvetica 30 bold ").place(relx=0.4, rely=0.7, relwidth=0.2, relheight=0.1)

def gender_recognition(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame,"Gender Prediction",gender_recognition)
    gender_recognition_frame = Frame(root, bg="white")
    gender_recognition_frame.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)
    toolbar_and_menu(gender_recognition_frame, "Gender Prediction")

    ImageLabel(gender_recognition_frame, image=get_gui_image("submit"), relx=0.03, rely=0.2)

    sent = Entry(gender_recognition_frame, font="Helvetica 30 bold ")
    sent.place(relx=0.31, rely=0.21, relwidth=0.35, relheight=0.1)
    ImageButton(gender_recognition_frame, image=get_gui_image("sentiment12"),
                    second_image=get_gui_image("sentiment12"),
                    bg="white", font="Helvetica 18 bold", bd=0,
                    command=lambda: predict_gender(sent.get(), gender_recognition_frame)).place(relx=0.31, rely=0.35)


def predict_gender(sent,gender_recognition_frame):
    genderr = gender.analyzer(sent)
    Label(gender_recognition_frame, text=genderr, bg="white", font="Helvetica 30 bold ").place(relx=0.4, rely=0.7,
                                                                                            relwidth=0.2, relheight=0.1)


def titanic_survivor(old_frame, check_call=None):
    if check_call is None:
        second_menu(old_frame, "Titanic Survivor", titanic_survivor)
    titanic_survivor_frame= Frame(root , bg="white")
    titanic_survivor_frame.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)
    toolbar_and_menu(titanic_survivor_frame, "Titanic Survivor")
    sent1 = Entry(titanic_survivor_frame)
    sent1.place(relx=0.4, rely=0.136, relwidth=0.2, relheight=0.08)
    sent2 = Entry(titanic_survivor_frame)
    sent2.place(relx=0.4, rely=0.276, relwidth=0.2, relheight=0.08)
    sent3 = Entry(titanic_survivor_frame)
    sent3.place(relx=0.4, rely=0.396, relwidth=0.2, relheight=0.08)
    sent4 = Entry(titanic_survivor_frame)
    sent4.place(relx=0.4, rely=0.516, relwidth=0.2, relheight=0.08)
    sent5 = Entry(titanic_survivor_frame)
    sent5.place(relx=0.4, rely=0.656, relwidth=0.2, relheight=0.08)
    sent6 = Entry(titanic_survivor_frame)
    sent6.place(relx=0.4, rely=0.796, relwidth=0.2, relheight=0.08)
    ImageButton(titanic_survivor_frame, image=get_gui_image("welcome_start_red"),
                second_image=get_gui_image("welcome_start_red"),
                bg="white", font="Helvetica 18 bold", bd=0,
                command=lambda: predict_survivor(sent1.get(),sent2.get(),sent3.get(),sent4.get(),
                                                 sent5.get(),sent6.get(),titanic_survivor_frame)).place(relx=0.7, rely=0.35)


    img = get_gui_image("titaniccc")
    Label(titanic_survivor_frame, image=img, bg="white", bd=0).place(relx=0.07, rely=0.13, relwidth=0.27,
                                                                 relheight=0.75)
    root.mainloop()

def predict_survivor(sent1,sent2,sent3,sent4,sent5,sent6,titanic_survivor_frame):
    titanicc=Titanic.survivor(sent1,sent2,sent3,sent4,sent5,sent6)
    Label(titanic_survivor_frame, text=titanicc,bg="white", font="Helvetica 30 bold ").place(relx=0.7, rely=0.75,
                                                                                            relwidth=0.19,
                                                                                            relheight=0.1)


def main_menu(old_frames):
    for old_frame in old_frames:
        if old_frame is None:
            continue
        old_frame.destroy()
    wid = GetSystemMetrics(0)
    hig = GetSystemMetrics(1)
    cross = get_gui_image("close_red_blue")
    cross2 = get_gui_image("close_red_blue_cross")
    mini = get_gui_image("mini_yellow_blue")
    mini2 = get_gui_image("mini_yellow_blue_min")
    full_frame = Frame(root, bg="#3838df").place(relx=0, rely=0, relwidth=1, relheight=1)
    ImageButton(full_frame, image=cross, second_image=cross2, command=close, border="0", bg="#3838df",
                bd=0).place(relx=0.982,
                            rely=0)
    ImageButton(full_frame, image=mini, second_image=mini2, command=minimize, border="0", bg="#3838df",
                bd=0).place(
        relx=0.962, rely=0)
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

    image = get_gui_image("Titanic Survivor")
    description_img = get_gui_image("titanic_survivor_about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: titanic_survivor(frame)).grid(row=1, column=0)

    image = get_gui_image("Face Classification")
    description_img = get_gui_image("face_recog_about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: facerecog(frame)).grid(row=1, column=1)

    image = get_gui_image("Emoji Predictor")
    description_img = get_gui_image("emoji_predict_about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: emojipredict(Menu_frame)).grid(row=1, column=2)

    image = get_gui_image("Flower Recognition")
    description_img = get_gui_image("flower_recog_about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: flower_recognition(Menu_frame)).grid(row=1, column=3)

    image = get_gui_image("Odd One Out")
    description_img = get_gui_image("odd one about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: odd_out(Menu_frame)).grid(row=2, column=0)

    image = get_gui_image("Word Analogy")
    description_img = get_gui_image("word analogy about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: word_analog(Menu_frame)).grid(row=2, column=1)

    image = get_gui_image("Spam or Ham")
    description_img = get_gui_image("spam or ham about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: spam_or_ham(Menu_frame)).grid(row=2, column=2)

    image = get_gui_image("Gender Prediction")
    description_img = get_gui_image("gender predict about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: gender_recognition(Menu_frame)).grid(row=2, column=3)

    image = get_gui_image("Character Recognition")
    description_img = get_gui_image("character recog about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: character_recognition(Menu_frame)).grid(row=3, column=0)

    image = get_gui_image("Sentiment Analysis")
    description_img = get_gui_image("senti anal about")
    ImageButton(Menu_frame, second_image=description_img, image=image, bg="white", border="0",
                command=lambda: sentiment_analysis(Menu_frame)).grid(row=3, column=1)

    Button(Menu_frame, text="Hello, Machine", bg="#2196f3", fg="#00CDF8", font="Helvetica 18 bold",
           command=lambda: main_menu([frame])).grid(row=6, column=1)


class ImageButton(Button):
    def __init__(self, master, second_image, image, **kw):
        Button.__init__(self, master=master, cursor="hand2", **kw)
        self['image'] = image
        self.default_image = image
        self.second_image = second_image
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self['image'] = self.second_image

    def on_leave(self, e):
        self['image'] = self.default_image


class ImageLabel(Frame):
    def __init__(self, frame, image, relx, rely):
        # render = get_gui_image(image_path)
        self.img = Label(frame, image=image, bg="white")
        self.img.image = image
        self.img.place(relx=relx, rely=rely)

    def Destroy(self):
        self.img.destroy()


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
        toaster = ToastNotifier()
        global Face_Recognition, Face_Record, odd_one_out, word_analogy, sentiment, Titanic, Spam
        sys.path.insert(1, "Face_Recog/")
        sys.path.insert(4, "odd_one_out/")
        sys.path.insert(5, "Word_analogy/")
        sys.path.insert(7, "Sentiment_Analysis/")
        #sys.path.insert(8, "Titanic_Survivor/")

        sentiment = __import__('sentiment',globals())
        sys.path.insert(6, "Spam_or_Ham/")

        #Titanic = __import__('Titanic', globals())
        Face_Recognition = __import__('Face_Recognition', globals())
        Face_Record = __import__('Face_Record', globals())
        # odd_one_out = __import__('odd_one_out', globals())
        # word_analogy = __import__('word_analogy', globals())
        # sentiment = __import__('sentiment', globals())
        # Spam = __import__('Spam', globals())
        # glove = open("Files/Emoji_Predictor/glove.6B.50d.txt", encoding='utf-8')
        # for line in glove:
        #     value = line.split()
        #     word = value[0]
        #     coefficient = np.asarray(value[1:], dtype=float)
        #     glove_dictionary[word] = coefficient
        # toaster.show_toast("Odd One Out is ready to run")
        # global word_vector
        # # word_vector = KeyedVectors.load_word2vec_format(
        # #     "Files/Word_Analogy/GoogleNews-vectors-negative300.bin", binary=True)
        # toaster.show_toast("Odd One Out and Word Analogy is ready to run")

def close():
    sys.exit(0)


def minimize():
    root.iconify()



if __name__ == "__main__":


    obj = A()
    obj.start()
    glove_dictionary = {}

    # word_vector = KeyedVectors.load_word2vec_format(
    #     " G:\Project\Fun With ML\Code\odd_one_out\Files\GoogleNews-vectors-negative300.bin", binary=True)

    root = Tk()
    root.title("Fun With ML & DL")

    root.attributes("-fullscreen", True)

    # root.attributes("-fullscreen", True)
    # root.attributes("-alpha", 0.55)


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

    img = get_gui_image("welcome_start_red")
    start = ImageButton(frame, second_image=img, image=img, command=lambda: main_menu([frame]), border="0", bg="white")
    img = get_gui_image("welcome_about_red")
    start.place(relx=0.05, rely=0.7)
    welcome_about = ImageButton(frame, second_image=img, image=img, command=about, border="0", bg="white").place(
        relx=0.77, rely=0.03)
    img = get_gui_image("welcome_img")
    Label(frame, image=img, bg="white").place(relx=0.5, rely=0.3, relwidth=0.5, relheight=0.7)

    root.mainloop()
