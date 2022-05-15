from tkinter import *
root=Tk()
e=Entry()
e.pack()
def fun1():
    label1=Label(root,text="you clicked the button",fg='green',bg='yellow')
    label1.pack()
button1=Button(root,text="Clik here",padx=10,pady=5,command=fun1,fg='blue',bg='pink')
button1.pack()
root.mainloop()