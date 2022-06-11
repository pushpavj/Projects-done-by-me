from flask import Flask, render_template, request, jsonify
#flask is library which help us create app or router for the app
from flask_cors import CORS, cross_origin
#flask cors is used when you deploy a code in one region example in india and you wanted to access
# the app from another geographycal region then flask_cors will help in transaction connections.
# For local  deployment flask_cors is not necessary.
import requests
from bs4 import BeautifulSoup as bs
#to beutify the html code in text list
from urllib.request import urlopen as uReq
# to open and read the url page we need urllib

app = Flask(__name__) #this is creating Flask app to access the function through route

#we need a route to access the function or methods from an another system flask establishes the
# connection between the other system to the methods and functions inside the python code through
# route or API
# In this code there are 2 functions one is homePage() and another one is index()


@app.route('/',methods=['GET'])  #route to display the home page
#app.route with only '/' is called root or base url i.e. if we hit the base url then
# it will execute the function below. Base url is local host followed by port number
@cross_origin() #required for global deployment such as cloud not required for local deployment
def homePage():
    return render_template('index.html') #render template to expose html code i.e it is going to
                                        #execute code in the index.html
                                    # indes.html should be available in templates folder
#the name of the folder should be 'templates' only other wise you need to provide entier path of the
# html file.


@app.route('/review',methods=['POST',"GET"])
@cross_origin()
def index():
    if request.method=='POST':
        try:
            searchString = request.form['content'].replace(" ","")
            flipkart_url='https://www.flipkart.com/search?q='+searchString
            uClient=uReq(flipkart_url)
           # print(flipkart_url)
            flipkartPage=uClient.read()
            uClient.close()
            flipkart_html=bs(flipkartPage,'html.parser')
            bigboxes=flipkart_html.findAll('div',{'class':'_1AtVbE col-12-12'})
            del bigboxes[0:3]
            box=bigboxes[0]
            productLink="https://www.flipkart.com"+box.div.div.div.a['href']
            prodRes=requests.get(productLink)
            prodRes.encoding = 'utf-8'
            prod_html = bs(prodRes.text,'html.parser')
          #  print(prod_html)
            commentboxes= prod_html.find_all('div',{'class','_16PBlm'})

            filename= searchString + '.csv'
            fw= open(filename,'w')
            headers = 'Product, Customer Name, Rating, Heading, Comment \n'
            fw.write(headers)
            reviews=[]

            for commentbox in commentboxes:
                try:
                    name=commentbox.div.div.find_all('p',{'class':'_2sc7ZR _2V5EHH'})[0].text
                except:
                    name = 'No name'

                try:
                    rating=commentbox.div.div.div.div.text
                except:
                    rating = "No rating"

                try:
                    commentHead=commentbox.div.div.div.p.text
                except:
                    commentHead="No comment heading"


                try:
                    comtag = commentbox.div.div.find_all('div', {'class': ''})
                    custComment = comtag[0].div.text
                except:
                    custComment = 'No Customer Comment'
                mydict={"Product":searchString,'Name':name, "Rating":rating,"Comment Head": commentHead,
                        "Comment":custComment}
                print(mydict)
                reviews.append(mydict)
            return render_template('results.html', reviews=reviews[0:(len(reviews)-1)])
           # return render_template('results.html', reviews=reviews[0:-1])
        except Exception as e:
            print('The exception message is ', e)
            return 'some thing is wrong'
    #else:
       # return render_template('index.html')

if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8000,debug=True) # running the app on the local machine on port 8000
    app.run(debug=True)

