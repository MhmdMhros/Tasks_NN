from tkinter import *
from model import *

window = Tk()
window.title("BachPropagation Task")
window.geometry("600x500")

function_type = StringVar()
num_layers = IntVar()
num_neurons = IntVar()
eta = DoubleVar()
m = IntVar()
isBias = IntVar()


def calculate_operation():
    train_model(function_type.get(), num_layers.get(), num_neurons.get(), eta.get(), m.get(), isBias.get())


user_input_frame = Frame(window, bg='lightblue')
# ================= A_Function_Choice ==================
# choice
function_type.set("sigmoid")  # Default selection
# choice between Sigmoid and Tanh
# choice1
option1_radio = Radiobutton(user_input_frame, text="Sigmoid", variable=function_type, value="sigmoid")
# choice 2
option2_radio = Radiobutton(user_input_frame, text="Tanh", variable=function_type, value="tanh")
# number of hidden layers
num_layers_label = Label(user_input_frame, text="#Layers", font=('Arial', 10))
num_layers_entry = Entry(user_input_frame, textvariable=num_layers, width=50)
# number of neurons in each hidden layer
num_neurons_label = Label(user_input_frame, text="#Neurons", font=('Arial', 10))
num_neurons_entry = Entry(user_input_frame, textvariable=num_neurons, width=50)
# learning rate (eta)
eta_label = Label(user_input_frame, text="Learning rate (eta)", font=('Arial', 10))
eta_entry = Entry(user_input_frame, textvariable=eta, width=50)
# number of epochs (m)
m_label = Label(user_input_frame, text="#Epochs (m)", font=('Arial', 10))
m_entry = Entry(user_input_frame, textvariable=m, width=50)
# bias check (exist or not exist 1 or 0)
bias_check = Checkbutton(user_input_frame, text="Add Bias", variable=isBias)
# Calculate button for calculate the operation and return the result
calc_button = Button(window, text='Calculate', command=calculate_operation, activeforeground='red', width=50, bg='lightblue')

option1_radio.grid(row=0, columnspan=5)
option2_radio.grid(row=1, columnspan=5)
num_layers_label.grid(row=2, column=0, pady=10)
num_layers_entry.grid(row=2, column=1, pady=10)
num_neurons_label.grid(row=3, column=0, pady=10)
num_neurons_entry.grid(row=3, column=1, pady=10)
eta_label.grid(row=4, column=0, pady=10)
eta_entry.grid(row=4, column=1, pady=10)
m_label.grid(row=5, column=0, pady=10)
m_entry.grid(row=5, column=1, pady=10)
bias_check.grid(row=6,columnspan=5, pady=10)

user_input_frame.grid(row=0, padx=90, pady=20)
calc_button.grid(row=1)

window.mainloop()
