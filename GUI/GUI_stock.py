import os
import tkinter
from tkinter import *

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# matplotlib reference : https://www.quora.com/How-do-I-create-a-real-time-plot-with-matplotlib-and-Tkinter

import pandas as pd
import numpy as np
import pickle

from core.data_processor import DataLoader

class main:
    def __init__(self, master):
        self.master = master
        
        Label(master,text="LSTM Predicting S&P500 price",fg="black",font="Times 20 bold").pack(pady=10)

        Button(master,font="Times 15",text="load recent data", command=self.getAndDrawData).pack()       
        Button(master,font="Times 15",text="point by point predict", command=self.pbpPredict).pack()
        Button(master,font="Times 15",text="multi sequence predict", command=self.msPredict).pack()
        Button(master,font="Times 15",text="clean", command=self.clear).pack()

        # get the data, using test data just for now
        # will use real new data instead of test data in the future
        self.dataloader = DataLoader("sp500.csv", 0.85, ["Close","Volume"])
        self.x_test, self.y_test = self.dataloader.get_test_data(seq_len=50, normalise=True)
        print("Successfully got data")       
        
        # show the prediction
        self.pr = Label(master,text="MSE Loss : ",fg="blue",font="Times 20 bold")
        self.pr.pack(pady=20)
        
    def clear(self):
        self.graph.get_tk_widget().destroy()
        print("clear!")
        self.pr['text'] = "MSE Loss : "


    def getAndDrawData(self):
        # to be implemented : retrieve recent S&P 500 data for predict, and then draw it
        # now : displaying test data
        print("yoyo")
        x_test_true, self.y_test_true =self.dataloader.get_test_data(seq_len=50, normalise=False)

        self.fig = Figure(figsize=(8, 7), facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.y_test_true, label='True Data')
        self.ax.grid()
 
        self.graph = FigureCanvasTkAgg(self.fig, master=self.master)
        self.graph.get_tk_widget().pack(side="top",fill='both',expand=True)
        self.graph.draw()

    def pbpPredict(self):
        # point by point predict
        with open("model_PBP_v1.pkl", 'rb') as file:  
            model = pickle.load(file)
        print("loaded PBP model")
        self.predictions = model.predict_point_by_point(self.x_test)

        predictions_true = []
        for i, p in enumerate(self.predictions):
            predictions_true.append((1+p) * self.dataloader.data_test[i][0])
        
        self.ax.plot(predictions_true, label='Point by Point Prediction')
        self.ax.grid()
        self.graph.draw()
            
        loss = self.calculateLoss(self.y_test_true, predictions_true, "PBP")
        self.pr['text'] = "MSE Loss :" + str(loss)

    def msPredict(self):
        # multi sequence predict
        # load the model
        with open("model_MS_v1.pkl", 'rb') as file:  
            model = pickle.load(file)
        print("loaded MS model")
        self.predictions = model.predict_sequences_multiple(self.x_test, 50,50)
        print("done makine multi-sequence predictions")
        predictions_true = self.dataloader.denormalise_windows(self.dataloader.data_test, self.predictions, False)

        # seperate each sequence
        for i, p in enumerate(predictions_true):
            padding = [None for _ in range(i * 50)]
            self.ax.plot(padding+p, label='Multi-Sequence Prediction')
            self.ax.grid()
            self.graph.draw()
            
        loss = self.calculateLoss(self.y_test_true, predictions_true, "MS")
        self.pr['text'] = "MSE Loss :" + str(loss)
        return

    def calculateLoss(self, true, pred, mode):
        sigma = 0
        # to be update : range
        print("len true ==", len(true))
        if mode == "MS":
            for i in range(650):
                sigma += (true[i][0] - pred[i//50][i%50]) ** 2
            return sigma / len(true)
        elif mode=="PBP":
            for i in range(650):
                sigma += (true[i][0] - pred[i]) ** 2
            return sigma / len(true)
        

if __name__ == "__main__":
    root = Tk()
    main(root)
    root.title('stock price predictor')
    root.resizable(0, 0)
    root.mainloop()
