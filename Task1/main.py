import tkinter
from tkinter import *
from tkinter import messagebox
from model import *

win = Tk()
win.geometry("500x500")


def calculate_operation(*args):
    if (algo_type.get() == '1') | (algo_type.get() == '2'):
        train_model(algo_type.get(), int(feature1.get()), int(feature2.get()), int(class1.get()), int(class2.get()),
                    eta.get(), int(m.get()), mse.get(), isBias.get())
    else:
        messagebox.showerror(title='Error', message='Choose a learning algorithm!')


def toggle_visibility_on(*args):
    mse_label.place_forget()
    mse_entry.place_forget()


def toggle_visibility_off(*args):
    mse_label.place(x=140, y=400)
    mse_entry.place(x=280, y=400)


algo_type = StringVar()
feature1 = StringVar()
feature2 = StringVar()
class1 = StringVar()
class2 = StringVar()
eta = DoubleVar()
m = StringVar()
mse = DoubleVar()
isBias = IntVar()

# region WIDGETS

perceptron_radio = Radiobutton(win, text='Perceptron', variable=algo_type, value=1, command=toggle_visibility_on)
perceptron_radio.place(x=140, y=45)

adaline_radio = Radiobutton(win, text='Adaline', variable=algo_type, value=2, command=toggle_visibility_off)
adaline_radio.place(x=280, y=45)

feature1_label = Label(win, text='Feature 1:')
feature1_label.place(x=140, y=100)
feature1_entry = Entry(win, textvariable=feature1)
feature1_entry.place(x=280, y=100)

feature2_label = Label(win, text='Feature 2:')
feature2_label.place(x=140, y=150)
feature2_entry = Entry(win, textvariable=feature2)
feature2_entry.place(x=280, y=150)

class1_label = Label(win, text='Class 1:')
class1_label.place(x=140, y=200)
class1_entry = Entry(win, textvariable=class1)
class1_entry.place(x=280, y=200)

class2_label = Label(win, text='Class 2:')
class2_label.place(x=140, y=250)
class2_entry = Entry(win, textvariable=class2)
class2_entry.place(x=280, y=250)

eta_label = Label(win, text='Learning Rate (eta):')
eta_label.place(x=140, y=300)
eta_entry = Entry(win, textvariable=eta)
eta_entry.place(x=280, y=300)

epochs_label = Label(win, text='Epochs Number (m):')
epochs_label.place(x=140, y=350)
epochs_entry = Entry(win, textvariable=m)
epochs_entry.place(x=280, y=350)

mse_label = Label(win, text='MSE Threshold:')
mse_label.place(x=140, y=400)
mse_entry = Entry(win, textvariable=mse)
mse_entry.place(x=280, y=400)

bias_check = Checkbutton(win, text="Add Bias", variable=isBias)
bias_check.place(x=140, y=450)

calc_button = Button(win, text='Calculate', command=calculate_operation, activeforeground='red')
calc_button.place(x=280, y=450)

# endregion

win.mainloop()
