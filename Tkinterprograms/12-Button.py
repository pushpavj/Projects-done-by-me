from tkinter import *
root=Tk()
def fun1():
    label1=Label(root,text="you clicked the button")
    label1.pack()
button1=Button(root,text="Clik here",padx=10,pady=5,command=fun1)
button1.pack()
root.mainloop()