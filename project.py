import streamlit as st
import hashlib
import base64
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from twilio.rest import Client
from streamlit_lottie import st_lottie
import requests
import re
st.set_page_config(page_title="HNN",page_icon="logo1.png",layout="wide")
warnings.filterwarnings("ignore")
with st.sidebar:
    selected=option_menu("Menu",["Login","Home","Predictor"])
if 'count' not in st.session_state:
	st.session_state.count = 0
if (selected=="Home"):
        if st.session_state.count == 1:
            def add_bg_from_local(image_file):
                with open(image_file, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
                    background-size: cover
                }}
                </style>
                """,
                unsafe_allow_html=True
                )
            css = """
            body{
            background-image:url(img);
            background-size: cover;
            text-align:center;
            color:Gray;
            }
            [data-testid="stImage"]{
            padding-top:5px;
            margin-left: auto;
            margin-right: auto;
            height:30%;
            width:30%;
            }"""
            st.write(f'<style>{css}</style>', unsafe_allow_html=True)
            st.write('<span style="font-size:30px;color:white"><b>⚕️Health Network Navigator(HNN)⚕️</b></span>',unsafe_allow_html=True)
            st.image("logo1.png")
            st.write('<span style="font-size:30px;color:white"><b>🔍Modern way to predict the disease by provided symptoms and find the best doctor to cure🔎</b></span>',unsafe_allow_html=True) 
            add_bg_from_local('black_back.jpg')
            def load_lottieurl(url):
                r=requests.get(url)
                if r.status_code!=200:
                    return None
                return r.json() 
            lottie_coding=load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_tul1qoqd.json")
            st_lottie(lottie_coding,height=300,key="coding") 
        else:
           st.warning("please login")    
        
if (selected=="Predictor"):
    if st.session_state.count == 1:
        bg="""
        <style>
        [data-testid="stAppViewContainer"]>.main{
        background-image: linear-gradient(to right bottom,#FFFFFF,#87CEEB);
        background-attachment:local;
        color:#FFFFFF;
        }
        </style>"""
        title="""
        <style>
        [class="css-zt5igj e16nr0p33"]{
            color:black;
            text-align:center;
            font-weight:bold; 
            font-size:80px; 
            }
        </style>"""
        Button="""
        <style>
        [class="css-1x8cf1d edgvbvh5"]{
            width:200px;
            height:50px;
            color:white;
            background-color:green;
            text-align:center;
            display:block;
            margin: 0 auto;
            }
        </style>"""
        btext="""
        <style>
        [data-testid="stMarkdownContainer"]{
        font-size:50px;
        }
        </style>"""
        sb="""
        <style>
        [class="css-vk3wp9 e1fqkh3o11"]{
        background-image: linear-gradient(to right bottom,#FFC0CB,#FFFFFF);
        }
        </style>"""
        hide="""
        <style>
        #MainMenu{visibility:hidden;}
        footer {visibility:hidden;}
        header{visibility:hidden;}
        </style>"""
        st.markdown(hide,unsafe_allow_html=True)
        st.markdown(title,unsafe_allow_html=True)
        st.markdown(bg,unsafe_allow_html=True)
        st.markdown(Button,unsafe_allow_html=True)
        st.markdown(btext,unsafe_allow_html=True)
        st.markdown(sb,unsafe_allow_html=True)
        st.title('Health Network Navigator')
        html_code = """
        <b><marquee Scrollamount="20" behavior="alternate" loop="" height="40px" style="font-family:Verdana;font-size:20px;color:red;background-color:black;">Symptoms <span style="color:white">Pettu </span>Doctor <span style="color:white">Pattu</span></marquee></b>
        """
        st.write(html_code, unsafe_allow_html=True)
        data=pd.read_csv('class.csv')
        data=data.replace(np.nan,0)
        d={}
        for i in data.columns:
            a=np.array(data[i]).tolist()
            a=[i for i in a if i!=0]
            d[i]=a
        data2=pd.read_csv('Testing.csv')
        placeholder=st.empty()    
        with placeholder.form("entry_form",clear_on_submit=True):
            col=[]*3
            col=st.columns(3)
            k=0
            a={}
            for i in data2.columns[:-1]:
                a[i]=0
            for i in d.keys():
                if k>=3:k=0
                with col[k]:
                    st.write(f'<span style="color:pink;font-size:25px"><b>{i.capitalize()}</b></span>',unsafe_allow_html=True)
                    for j in d[i]:
                        b=st.checkbox(j.capitalize())
                        if b: 
                          a[j]=1   
                k+=1      
            b=st.form_submit_button("Predict & Navigate")
        def model(a):
            train_df = pd.read_csv("Training.csv")
            x_train = train_df[train_df.columns[:-1]]
            y_train = train_df[train_df.columns[-1]]
            rfc_mdl = RandomForestClassifier(n_estimators = 100,n_jobs = -1)
            nb_clf = GaussianNB()
            svm_clf = SVC()
            models = [("RandomForestClassifier",rfc_mdl),("NaviesBayes",nb_clf),("SupportVectorMachine",svm_clf)]
            stacking = StackingClassifier(estimators = models,final_estimator=LogisticRegression(),n_jobs = -1)
            stacking.fit(x_train,y_train)
            arr = np.array(np.array(a))
            c=stacking.predict(arr.reshape(1,-1))
            return c 
        if b:
            if all(ele == 0 for ele in list(a.values())):
                placeholder.empty() 
                st.write('<span style="font-size: 30px;"><b>Congrats <span style="color:pink;">❤</span> you are Healthy</b></span>',unsafe_allow_html=True)
            else:
                with st.spinner('Results Loading...'):
                    res=model(list(a.values())) 
                str1=''
                for i in a.keys():
                    if a[i]==1:str1=str1+i+' , '
                data1=pd.read_csv("doctors4.csv")
                data4=pd.read_csv("precaution.csv")
                link=data1[data1["Prognosis"]==res[0]]["Map_Link"]
                name=data1[data1["Prognosis"]==res[0]]["Specailist"]
                hospital=data1[data1["Prognosis"]==res[0]]["Hospital"]
                precaution=data4[data4["prognosis"]==res[0]]["Introduction"]
                remedy=data4[data4["prognosis"]==res[0]]["Remedies"] 
                placeholder.empty()    
                st.write(f'<span style="font-size:20px;color:green;">You may have <span style="font-size:25px;color:black;"><b>{res[0].capitalize()}</b></span> based on given symptoms <span style="color:red;">{str1[:-2]}</span></span>',unsafe_allow_html=True)
                st.write(f'<br><span style="font-size: 25px;color:black;"><b>{res[0].capitalize()}: </b></span><span style="font-size: 20px;color:green;">{precaution.tolist()[0]}</span>',unsafe_allow_html=True)
                st.write(f'<br><span style="font-size: 25px;color:black;"><b>The following remedies might help you:</b></span><br><span style="font-size: 20px;color:green;">{remedy.tolist()[0]}</span>',unsafe_allow_html=True)
                for i in range(len(link.tolist())):
                        st.write(f"{''.join(['-' for i in range(10)])}")
                        st.write(f'<span style="font-size: 20px;color:green;">You can consult this doctor: <span style="font-size:25px;color:black;"><b>{name.tolist()[i]}</b></span><br>at <span style="font-size:25px;color:black;"><b>{hospital.tolist()[i]}</b></span><br>for hospital address tap link <b><a href={link.tolist()[i]}><span style="font-size:25px;color:black;">Map_Link</span></a></b></span>',unsafe_allow_html=True)
                
                        
    else:
        st.warning("please login")

if (selected=="Login"):
    sb="""
    <style>
    [class="css-vk3wp9 e1fqkh3o11"]{
     background-image: linear-gradient(to right bottom,pink,violet,#87cefa);
    }
    </style>"""
    log="""
    <style>
    [class="main css-uf99v8 egzxvld5"]{
        background-image: linear-gradient(to right bottom,#FFFFFF,#87CEEB);
    }
    </style>"""
    st.markdown(log,unsafe_allow_html=True)
    st.markdown(sb,unsafe_allow_html=True)
    def make_hashes(password):
        return hashlib.sha256(str.encode(password)).hexdigest()

    def check_hashes(password,hashed_text):
        if make_hashes(password) == hashed_text:
            return hashed_text
        return False
    import sqlite3 
    conn = sqlite3.connect('data2.db')
    c = conn.cursor()
    # DB  Functions
    def create_usertable():
        c.execute('CREATE TABLE IF NOT EXISTS user1table(username TEXT,password TEXT,email TEXT PRIMARY KEY,dob DATE,phone TEXT)')
    def add_userdata(username,password,email,dob,phone):
        c.execute('INSERT INTO user1table(username,password,email,dob,phone) VALUES (?,?,?,?,?)',(username,password,email,dob,phone))
        conn.commit()

    def login_user(email,password):
        c.execute('SELECT * FROM user1table WHERE email =? AND password = ?',(email,password))
        data = c.fetchall()
        return data
    def already(email,phone):
        c.execute('SELECT * FROM user1table WHERE email =? OR phone= ?',(email,phone))
        data = c.fetchall()
        return data
    def view_all_users():
        c.execute('SELECT * FROM user1table')
        data = c.fetchall()
        return data
    import datetime
    select=st.selectbox("Login/Signup",options=("Login","Signup"))
    if select=="Login":
        st.subheader("Login Section")
        mail = st.text_input("Email")
        password = st.text_input("Password",type='password')
        if st.button("Login"):
                create_usertable()
                hashed_pswd = make_hashes(password)
                result = login_user(mail,check_hashes(password,hashed_pswd))
                if result:
                   st.balloons()
                   st.success("login successfull")
                   st.session_state.count =1
                else:
                    st.session_state.count =0
                    st.error("Incorrect Username/Password")
    if select=="Signup":
        st.subheader("Create New Account")
        new_user = st.text_input("Name")
        new_password = st.text_input("Strong_Password",type='password')
        new_email = st.text_input("Email")
        new_dob = st.date_input("When\'s your birthday",datetime.date(2000,1,1))
        new_phone=st.text_input("Phone",max_chars=10)
        regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
        if st.button("Signup"):
            if len(new_user)<6 or len(new_password)<6 or not re.fullmatch(regex,new_email) or len(new_phone)<10:
                st.error('please fill details correctly')
            else:   
                if already(new_email,new_phone):
                    st.warning('Already had an account/please login')
                else:    
                    create_usertable()
                    add_userdata(new_user,make_hashes(new_password),new_email,new_dob,new_phone)
                    st.success("You have successfully created a valid Account")
                    st.info("Go to Login Menu to login")
                       
