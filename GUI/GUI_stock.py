import os
import tkinter
from tkinter import *
import tkinter.filedialog as filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
# matplotlib reference : https://www.quora.com/How-do-I-create-a-real-time-plot-with-matplotlib-and-Tkinter

import pandas as pd
import numpy as np
import pickle

from core.data_processor import DataLoader

class main:
    def __init__(self, master):
        self.master = master
        
        Label(master,text="LSTM Predicting S&P500 price",fg="black",font="Times 20 bold").pack(pady=10)
        Button(master,font="Times 15",text="load historical data", command=self.openfile).pack()
        
        self.scale1 = Scale(master, orient = HORIZONTAL, label="used data len:", from_=50, to=650, resolution=50)
        self.scale1.pack(anchor = CENTER)
        
        self.scale2 = Scale(master, orient = HORIZONTAL, label="predict len:", from_=25, to=50, resolution=25)
        self.scale2.pack(anchor = CENTER)
        
         
        
        
        
        
        Button(master,font="Times 15",text="point by point predict", command=self.pbpPredict).pack()
        Button(master,font="Times 15",text="multi sequence predict", command=self.msPredict).pack()
        Button(master,font="Times 15",text="clean", command=self.clear).pack()
        
        # show the performance
        self.pr = Label(master,text="MAE Loss :    , MSE Loss :",fg="blue",font="Times 20 bold")
        self.pr.pack(pady=20)

        # plot empty figure
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        
        self.graph = FigureCanvasTkAgg(self.fig, master=root)
        self.graph.get_tk_widget().pack(side="top",fill='both',expand=True)
        
    def clear(self):
        self.ax.cla()
        self.ax.grid()
        self.graph.draw()
        print("clear!")
        self.pr['text'] = "MAE Loss :    , MSE Loss :"

    def openfile(self):
        # read csv file
        self.filename = filedialog.askopenfilename()
        self.dataloader = DataLoader(self.filename, ["Close","Volume","Open", "High","Low","Rocr100","Plus_dm"])

        print(self.filename)
        # load true data without normalise
        seq_len = int(self.scale2.get())
        x_test_true, self.y_test_true =self.dataloader.get_test_data(seq_len, normalise=False)
        print("load the data successfully")  

        # make sure the data len is valid
        self.scale1.configure(to=self.dataloader.len_test - 50)
        self.scale1.set(self.dataloader.len_test - 50)
        self.ax.cla()
        self.ax.grid()
        self.ax.plot(self.y_test_true, label='True Data')
        self.graph.draw()
    
    def plotTrueData(self):
        self.clear()
        self.ax.grid()
        self.ax.plot(self.y_test_true[:self.scale1.get()], label='True Data')
        self.graph.draw()
        print("update custom data successfully")  
        
        

    def pbpPredict(self):
        # point by point predict
        seq_len = int(self.scale2.get())
        # load the model based on the asked predict len
        if seq_len == 50:
            with open("models/improved_PBP_50.pkl", 'rb') as file:  
                model = pickle.load(file)
        elif seq_len == 25:
            with open("models/improved_PBP_25.pkl", 'rb') as file:  
                model = pickle.load(file)

        # reload data based on seq_len
        self.x_test, self.y_test = self.dataloader.get_test_data(seq_len, normalise=True)
        x_test_true, self.y_test_true =self.dataloader.get_test_data(seq_len, normalise=False)
        self.plotTrueData()
        
        print("successfully loaded point-by-point model")
        self.predictions = model.predict_point_by_point(self.x_test)
        print("done predicting using PBP model")

        # denormalize
        predictions_true = []
        for i, p in enumerate(self.predictions):
            predictions_true.append((1+p) * self.dataloader.data_test[i][0])

        self.ax.plot(predictions_true, label='Point by Point Prediction')
        self.graph.draw()

        loss = self.calculateMSELoss(self.y_test_true, predictions_true, "PBP")
        loss2 = self.calculateMAELoss(self.y_test_true, predictions_true, "PBP")
        self.pr['text'] = "MAE Loss :" + str(loss2) + ", MSE Loss :" + str(loss)


    def msPredict(self):
        # multi sequence predict
        # load the model
        seq_len = int(self.scale2.get())
        
        # load the model based on the asked predict len
        if seq_len == 50:
            with open("models/improved_MS_50.pkl", 'rb') as file:  
                model = pickle.load(file)
        elif seq_len == 25:
            with open("models/improved_MS_25.pkl", 'rb') as file:  
                model = pickle.load(file)
        print("succefully loaded MS model")

        # reload data based on seq_len
        self.x_test, self.y_test = self.dataloader.get_test_data(seq_len, normalise=True)
        x_test_true, self.y_test_true =self.dataloader.get_test_data(seq_len, normalise=False)

        # make predictions with len=seq_len
        self.predictions = model.predict_sequences_multiple(self.x_test, seq_len,seq_len)
        predictions_true = self.dataloader.denormalise_windows(seq_len, self.dataloader.data_test , self.predictions, False)

        self.plotTrueData()
        self.ax.grid()
        # seperate each sequence
        for i, p in enumerate(predictions_true):
            if i * seq_len >= int(self.scale1.get()):
                break
            padding = [None for _ in range(i * seq_len)]
            self.ax.plot(padding+p, label='Multi-Sequence Prediction')
            self.graph.draw()
        
        loss = self.calculateMSELoss(self.y_test_true, predictions_true, "MS")
        loss2 = self.calculateMAELoss(self.y_test_true, predictions_true, "MS")
        self.pr['text'] = "MAE Loss :" + str(loss2) + ", MSE Loss :" + str(loss)
        

    def calculateMSELoss(self, true, pred, mode):
        sigma = 0
        custom_len = int(self.scale1.get())
        seq_len = int(self.scale2.get())
        if mode == "MS":
            for i in range(custom_len):
                sigma += (true[i][0] - pred[i//seq_len][i%seq_len]) ** 2
            return sigma / custom_len
        elif mode=="PBP":
            for i in range(custom_len):
                sigma += (true[i][0] - pred[i]) ** 2
            return sigma / custom_len

    def calculateMAELoss(self, true, pred, mode):
        sigma = 0
        custom_len = int(self.scale1.get())
        seq_len = int(self.scale2.get())
        if mode == "MS":
            for i in range(custom_len):
                sigma += abs(true[i][0] - pred[i//seq_len][i%seq_len])            
            return sigma / custom_len
        elif mode=="PBP":
            for i in range(custom_len):
                sigma += abs(true[i][0] - pred[i])
            return sigma / custom_len
        

if __name__ == "__main__":
    root = Tk()
    main(root)
    root.title('stock price predictor')
    root.resizable(0, 0)
    root.mainloop()
