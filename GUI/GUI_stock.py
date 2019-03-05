import os
import tkinter
from tkinter import *

class main:
    def __init__(self, master):
        self.master = master
        self.c = Canvas(self.master,bd=3,relief="ridge", width=450, height=300, bg='white')
        self.c.pack(side=LEFT)
        
        Label(master,text="LSTM Predicting S&P500 price",fg="black",font="Times 20 bold").pack(pady=10)

        Button(master,font="Times 15",fg="white",bg="red", text="load recent data", command=self.getAndDrawData).pack()       
        Button(master,font="Times 15",fg="white",bg="red", text="point by point predict", command=self.pbpPredict).pack()
        Button(master,font="Times 15",fg="white",bg="red", text="multi sequence predict", command=self.msPredict).pack()
        Button(master,font="Times 15",fg="white",bg="red", text="clean", command=self.clear).pack()

        # show the prediction
        self.pr = Label(master,text="Prediction: ",fg="blue",font="Times 20 bold")
        self.pr.pack(pady=20)
        
        # x axis and y axis
        self.c.create_line([10, 295, 400, 295], fill='black')
        self.c.create_text([415, 295],text="time")
        self.c.create_line([10, 295, 10, 60], fill='black')
        self.c.create_text([20, 50],text="price")

        
    def clear(self):
        self.c.delete('all')
        self.pr['text'] = "Prediction: "

        # resume x axis and y axis
        self.c.create_line([10, 295, 400, 295], fill='black')
        self.c.create_text([415, 295],text="time")
        self.c.create_line([10, 295, 10, 60], fill='black')
        self.c.create_text([20, 50],text="price")


    def getAndDrawData(self):
        # to be implemented : retrieve recent S&P 500 data for predict, and then draw it
        self.open_price = [2713, 2727, 2722, 2720, 2710, 2715, 2757, 2780, 2753, 2715]
        self.high_price = [2731, 2754, 2737, 2747, 2731, 2747, 2780, 2789, 2761, 2730]
        self.low_price = [2689, 2725, 2706, 2701, 2697, 2713, 2753, 2744, 2713, 2659]
        self.close_price = [2731, 2732, 2716, 2701, 2703, 2747, 2779, 2744, 2713, 2677]
        #volume = []

        
        for p in [self.open_price, self.high_price, self.low_price, self.close_price]:
            # scale the price to make it fit in canvas
            p[:] = list(map(lambda x: 2900 - x, p))
            for i in range(10):
                # insert x coordinate for drawing
                p.insert(2*i, 20+30*i)

        # draw the lines and labels for each price
        self.c.create_line(self.open_price, fill='red')
        self.c.create_line(self.high_price, fill='green')
        self.c.create_line(self.low_price, fill='blue')        
        self.c.create_line(self.close_price, fill='cyan')

        self.c.create_text(self.open_price[-2]+30, self.open_price[-1],text="open price")
        self.c.create_text(self.high_price[-2]+30, self.high_price[-1]-5,text="high price")
        self.c.create_text(self.low_price[-2]+30, self.low_price[-1]+5,text="low price")
        self.c.create_text(self.close_price[-2]+30, self.close_price[-1],text="close price")

    def pbpPredict(self):
        # to be implemented : point by point predict
        predict = []
        for i in range(len(self.open_price)):
            if i % 2 == 0:
                predict.append(self.open_price[i]+20)
            else:
                predict.append((self.open_price[i] + self.high_price[i] + self.low_price[i] + self.close_price[i])/4)
        self.c.create_line(predict, fill='black')
        self.c.create_text(predict[-2]+60, predict[-1],text="point to point predict")

    def msPredict(self):
        # to be implemented : multi sequence predict
        self.c.create_arc(120, 150, 320, 350, style=ARC)
        self.c.create_text(370, 260,text="multi sequence predict")
        self.pr['text'] = "Prediction: decline"
        return

if __name__ == "__main__":
    root = Tk()
    main(root)
    root.title('stock price predictor')
    root.resizable(0, 0)
    root.mainloop()
