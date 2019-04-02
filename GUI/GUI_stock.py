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
        
        self.scale2 = Scale(master, orient = HORIZONTAL, label="prediction len:", from_=5, to=50, resolution=5)
        #self.scale2.pack(anchor = CENTER)
        
         
        
        
        
        
        Button(master,font="Times 15",text="point by point predict", command=self.pbpPredict).pack()
        Button(master,font="Times 15",text="multi sequence predict", command=self.msPredict).pack()
        Button(master,font="Times 15",text="clean", command=self.clear).pack()
        
        # show the prediction
        self.pr = Label(master,text="MSE Loss : ",fg="blue",font="Times 20 bold")
        self.pr.pack(pady=20)
        
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        
        self.graph = FigureCanvasTkAgg(self.fig, master=root)
        self.graph.get_tk_widget().pack(side="top",fill='both',expand=True)
        
        
        
    def clear(self):
        self.ax.cla()
        self.ax.grid()
        self.graph.draw()
        print("clear!")
        self.pr['text'] = "MSE Loss : "

    def openfile(self):
        # read csv file
        self.filename = filedialog.askopenfilename()
        self.dataloader = DataLoader(self.filename, 0.85, ["Close","Volume"])
    
        self.x_test, self.y_test = self.dataloader.get_test_data(seq_len=50, normalise=True)
        
        # load true data without normalise
        x_test_true, self.y_test_true =self.dataloader.get_test_data(seq_len=50, normalise=False)
        print("load the data successfully")  
        
        self.scale1.set(650)
        self.ax.cla()
        self.ax.grid()
        self.ax.plot(self.y_test_true, label='True Data')
        self.graph.draw()
    
    def setCustomData(self):
        self.clear()
        
        custom_len = int(self.scale1.get())
        seq_len = int(self.scale2.get())
        # todo : change 50 to seq_len
        seq_len = 50
        self.x_test, self.y_test = self.dataloader.get_custom_data(self.filename, seq_len, True, ["Close","Volume"], custom_len)
        
        # load true data without normalise
        x_test_true, self.y_test_true =self.dataloader.get_custom_data(self.filename, seq_len, False, ["Close","Volume"], custom_len)
        self.ax.grid()
        self.ax.plot(self.dataloader.true_full, label='True Data')
        self.graph.draw()
        print("update custom data successfully")  
        
        

    def pbpPredict(self):
        # point by point predict
        self.setCustomData()
        #seq_len = int(self.scale2.get())
        seq_len = 50
        with open("model_PBP.pkl", 'rb') as file:  
            model = pickle.load(file)
        print("successfully loaded point-by-point model")
        self.predictions = model.predict_point_by_point(self.x_test)
        print("done predicting using PBP model")
        predictions_true = [None] * seq_len
        for i, p in enumerate(self.predictions):
            predictions_true.append((1+p) * self.dataloader.data_test[i][0])

        self.ax.plot(predictions_true, label='Point by Point Prediction')
        self.graph.draw()
            
        loss = self.calculateLoss(self.y_test_true, predictions_true, "PBP")
        self.pr['text'] = "MSE Loss :" + str(loss)

    def msPredict(self):
        # multi sequence predict
        # load the model
        self.setCustomData()
        
        with open("model_MS.pkl", 'rb') as file:  
            model = pickle.load(file)
        print("succefully loaded MS model")
        #seq_len = int(self.scale2.get())
        seq_len = 50
        self.predictions = model.predict_sequences_multiple(self.x_test, 50,50)
        print("done predicting using multi-sequence model")
        predictions_true = self.dataloader.denormalise_windows(self.dataloader.data_test, self.predictions, False)

        # seperate each sequence
        for i, p in enumerate(predictions_true):
            padding = [None for _ in range((i+1) * seq_len)]
            self.ax.plot(padding+p, label='Multi-Sequence Prediction')
            self.graph.draw()
            
        loss = self.calculateLoss(self.y_test_true, predictions_true, "MS")
        self.pr['text'] = "MSE Loss :" + str(loss)
        

    def calculateLoss(self, true, pred, mode):
        sigma = 0
        # to be update : range
        custom_len = int(self.scale1.get())
        seq_len = int(self.scale2.get())
        seq_len = 50
        if mode == "MS":
            for i in range(custom_len-seq_len):
                sigma += (true[i][0] - pred[i//50][i%50]) ** 2
            return sigma / len(true)
        elif mode=="PBP":
            for i in range(custom_len-seq_len):
                sigma += (true[i][0] - pred[i+seq_len]) ** 2
            return sigma / len(true)
        

if __name__ == "__main__":
    root = Tk()
    main(root)
    root.title('stock price predictor')
    root.resizable(0, 0)
    root.mainloop()
