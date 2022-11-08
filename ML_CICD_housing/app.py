#from crypt import methods
from flask import Flask

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def index():
    return "Starting Machine Learning project _New message with CICD pipeline establishment"


if __name__=="__main__":
    app.run(debug=True)

