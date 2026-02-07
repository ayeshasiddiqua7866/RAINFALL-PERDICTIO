import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd
from tkinter import StringVar
from tkinter import *

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm
import Preprocess as pre
import NeuralNetwork as NN
import arima as fc


bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"


def Home():
        global window
        window = tk.Tk()
        window.title("RAINFALL Prediction and Forecasting Using CNN and ARIMA")
        C = tk.Canvas(window, bg="blue", height=250, width=300)
        filename = ImageTk.PhotoImage(file = "1.jpeg")
        background_label = tk.Label(window, image=filename)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        C.pack()
        def clear():
            print("Clear1")
            txt.delete(0, 'end')
            txt1.config(state = NORMAL)
            txt1.delete(0, 'end')
            txt1.config(state = "readonly")
            txt2.delete(0, 'end')
            
        def show():
                sname=clicked.get()
                txt1.config(state = NORMAL)
                txt1.delete(0, 'end')
                txt1.insert('end',sname)
                txt1.config(state = "readonly")
        
        # Dropdown menu options
        #co=['SUBDIVISION']
        df=pd.read_csv("data.csv")
        options =df['SUBDIVISION'].unique()
        # datatype of menu text
        clicked = StringVar()
        # initial menu text
        clicked.set( "KONKAN & GOA" )

        

 
        window.geometry('1280x720')
        window.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)

        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        

        message1 = tk.Label(window, text="RAINFALL Prediction and Forecasting Using LSTM and ARIMA" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
        message1.place(x=100, y=20)

        lbl = tk.Label(window, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl.place(x=100, y=200)
        
        txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt.place(x=400, y=215)
        lbl1 = tk.Label(window, text="Select Area",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1.place(x=100, y=300)
        drop = OptionMenu( window , clicked , *options)
        drop.place(x=400, y=315)
        lbl1 = tk.Label(window, text="Selected Area",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1.place(x=100, y=400)
        txt1 = tk.Entry(window,state="readonly",width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1.place(x=400, y=415)
        lbl1 = tk.Label(window, text="No.of Years to Forecast",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1.place(x=100, y=500)
        
        txt2 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt2.place(x=400, y=515)


        def browse():
                path=filedialog.askopenfilename()
                print(path)
                txt.delete(0, 'end')
                txt.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Train Dataset")      

        def preproc():
                sym=txt.get()
                if sym != "" :
                        pre.process(sym)
                        print("preprocess")
                        tm.showinfo("Input", "Preprocess Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset")

        def NNprocess():
                sym=txt.get()
                sname=txt1.get()
                noofyears=txt2.get()
                if sym != "" and sname!="" and noofyears!="" :
                        NN.process(sym,int(noofyears),sname)
                        tm.showinfo("Input", "Neural Network Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset")
        def forecast():
                sym=txt.get()
                sname=txt1.get()
                noofyears=txt2.get()
                if sym != "" and sname!="" and noofyears!="" :
                        fc.process(sym,sname,noofyears)
                        tm.showinfo("Input", "Forecasting Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset")
                
                

        browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse.place(x=650, y=200)

        clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
        clearButton.place(x=950, y=200)
        button = Button( window , text = "Set Selected Option" , command = show ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
        button.place(x=650, y=300)
         
        proc = tk.Button(window, text="Preprocess", command=preproc  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        proc.place(x=100, y=600)
        

        NNbutton = tk.Button(window, text="NEURAL NETWORK", command=NNprocess  ,fg=fgcolor   ,bg=bgcolor1 ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        NNbutton.place(x=400, y=600)
        forcast = tk.Button(window, text="ARIMA", command=forecast  ,fg=fgcolor   ,bg=bgcolor1 ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        forcast.place(x=700, y=600)

        quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=1000, y=600)

        window.mainloop()
Home()

