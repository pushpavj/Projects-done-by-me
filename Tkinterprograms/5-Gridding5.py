from tkinter import *
root=Tk()

label1=Label(root,text="Hello World")
label2=Label(root,text='This is Python')
label3=Label(root,text="                ")

label1.grid(row=0,column=0)
label2.grid(row=1,column=5)
label3.grid(row=0,column=2)
root.mainloop()