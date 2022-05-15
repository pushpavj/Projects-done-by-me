from tkinter import *
root=Tk()
e=Entry(root,width=50,border=5)
e.insert(0,"Enter your name")
e.pack()
def fun1():
    label1=Label(root,text="you entered as "+e.get(),fg='green',bg='yellow')
    label1.pack()
button1=Button(root,text="Clik here",padx=10,pady=5,command=fun1,fg='blue',bg='pink')
button1.pack()
root.mainloop()