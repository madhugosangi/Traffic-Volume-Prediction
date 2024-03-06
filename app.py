from flask import Flask,render_template,request
import pickle
import numpy as np
from numpy import asarray

app=Flask(__name__)
model=pickle.load(open('savemodel.sav','rb'))

@app.route("/")
def root():
    return render_template("train1.html")




@app.route('/predict',methods=['POST'])

def predict():
    if request.method=='POST':
        to_predict_list=request.form.to_dict()
        print(to_predict_list)

        to_predict_list=list(to_predict_list.values())
        new_data=asarray(to_predict_list)
       

        to_predict_list=list(map(int,new_data))
        print(to_predict_list)


        to_predict=np.array(to_predict_list).reshape(1,9)
        result=model.predict(to_predict)[0]
        print(result)

        if (result <=1000):
          prediction="No Traffic"
        elif(result >1000 and result <=3000):
          prediction="Busy or Normal flow"
        elif(result > 3000 and result <5500):
          prediction="Heavy Traffic"
        else:
           prediction="Worst Case"
           
        return render_template("result.html",prediction=prediction)
    
  




if __name__=='__main__':
    app.run(debug=True)
    app.config["TEMPLATES_AUTO_RELOAD"]=True