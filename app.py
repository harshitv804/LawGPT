from flask import Flask
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import os
import base64
from PIL import Image
from datetime import datetime
from datetime import date
import datetime
import random
from random import seed
from random import randint
import re
import PIL.Image
from PIL import Image
from flask import send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import threading
import time
import shutil
import hashlib
import urllib.request
import urllib.parse
from urllib.request import urlopen
import webbrowser
import json
import mysql.connector
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import urllib.request
import urllib.parse


import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from gensim.parsing.porter import PorterStemmer
#from keras.layers import Input, Dense, LSTM, TimeDistributed
#import spacy
#nlp = spacy.load('en')

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="lawyerbot"
)


app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####

@app.route('/',methods=['POST','GET'])
def index():
    msg=""
    mycursor = mydb.cursor()
    if request.method == 'POST':
        
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM cc_register where uname=%s && pass=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            #result=" Your Logged in sucessfully**"
            return redirect(url_for('bot')) 
        else:
            msg="You are logged in fail!!!"

    return render_template('index.html',msg=msg)

@app.route('/login',methods=['POST','GET'])
def login():
    cnt=0
    act=""
    msg=""
    if request.method == 'POST':
        
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM cc_admin where username=%s && password=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            #result=" Your Logged in sucessfully**"
            return redirect(url_for('admin')) 
        else:
            msg="You are logged in fail!!!"
        

    return render_template('login.html',msg=msg,act=act)

@app.route('/login_user',methods=['POST','GET'])
def login_user():
    cnt=0
    act=""
    msg=""
    if request.method == 'POST':
        
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM cc_register where uname=%s && pass=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            session['username'] = username1
            #result=" Your Logged in sucessfully**"
            return redirect(url_for('bot')) 
        else:
            msg="You are logged in fail!!!"
        

    return render_template('login_user.html',msg=msg,act=act)

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    msg=""
    mycursor = mydb.cursor()
    if request.method=='POST':
        file = request.files['file']

        fn="datafile.csv"
        file.save(os.path.join("static/upload", fn))

        filename = 'static/upload/datafile.csv'
        data1 = pd.read_csv(filename, header=0)
        data2 = list(data1.values.flatten())
        for ss in data1.values:
            val1=""
            if pd.isnull(ss[1]):
                val1=""
            else:
                val1=ss[1]
                
            val2=""
            if pd.isnull(ss[2]):
                val2=""
            else:
                val2=ss[2]
                
            val3=""
            if pd.isnull(ss[3]):
                val3=""
            else:
                val3=ss[3]

            
                
            '''mycursor.execute("SELECT max(id)+1 FROM cc_data")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1

            sql = "INSERT INTO cc_data(id,description,offense,punishment,section) VALUES (%s,%s,%s,%s,%s)"
            val = (maxid,ss[0],val1,val2,val3)
            mycursor.execute(sql, val)
            #mydb.commit()'''
        
        msg="success"
        
    return render_template('admin.html',msg=msg)

@app.route('/process', methods=['GET', 'POST'])
def process():
    msg=""
    cnt=0
    

    filename = 'static/upload/datafile.csv'
    data1 = pd.read_csv(filename, header=0)
    data2 = list(data1.values.flatten())

    
    data=[]
    i=0
    sd=len(data1)
    rows=len(data1.values)
    
    #print(str(sd)+" "+str(rows))
    for ss in data1.values:
        cnt=len(ss)
        data.append(ss)
    cols=cnt

    
    return render_template('process.html',data=data, msg=msg, rows=rows, cols=cols)

@app.route('/process2', methods=['GET', 'POST'])
def process2():
    msg=""
    act=request.args.get("act")
    
    return render_template('process2.html',msg=msg, act=act)

@app.route('/add_query', methods=['GET', 'POST'])
def add_query():
    msg=""
    sid=""
    mycursor = mydb.cursor()

    cnt=0
    
    data=[]
    

    mycursor.execute("SELECT * FROM cc_data")
    data = mycursor.fetchall()
        
    
    if request.method=='POST':
        sid=request.form['sid']
        
        msg="success"

    return render_template('add_query.html',msg=msg,sid=sid,data=data)


@app.route('/add_query1', methods=['GET', 'POST'])
def add_query1():
    msg=""
    act=request.args.get("act")
    sid=request.args.get("sid")
    mycursor = mydb.cursor()
    
    cnt=0
    #filename = 'static/upload/datafile.csv'
    #data1 = pd.read_csv(filename, header=0,encoding='cp1252')
    #data2 = list(data1.values.flatten())

    
    data=[]
    #i=0
    #for ss in data1.values:
    #    cnt=len(ss)
    #    data.append(ss)
    #cols=cnt

    mycursor.execute("SELECT * FROM cc_data where id=%s",(sid,))
    data = mycursor.fetchone()

    mycursor.execute("SELECT * FROM cc_contact")
    data2 = mycursor.fetchall()
        
    
    if request.method=='POST':
        
        user_query=request.form['user_query']
        district=request.form['district']
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        address=request.form['address']
        
        
        mycursor.execute("update cc_data set user_query=%s,district=%s,name=%s,mobile=%s,email=%s,address=%s where id=%s",(user_query,district,name,mobile,email,address,sid))
        mydb.commit()

        mycursor.execute("SELECT count(*) FROM cc_contact where id=%s && name=%s && mobile=%s",(sid,name,mobile))
        d1 = mycursor.fetchone()[0]
        if d1==0:
            mycursor.execute("SELECT max(id)+1 FROM cc_contact")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1

            

            sql = "INSERT INTO cc_contact(id,section_id,district,name,mobile,email,address) VALUES (%s,%s,%s,%s,%s,%s,%s)"
            val = (maxid,sid,district,name,mobile,email,address)
            mycursor.execute(sql, val)
            mydb.commit()
        else:
            mycursor.execute("update cc_contact set district=%s,name=%s,mobile=%s,email=%s,address=%s where section_id=%s && name=%s",(district,name,mobile,email,address,sid,name))
            mydb.commit()
        msg="success"

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from cc_contact where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_query1',sid=sid)) 

    return render_template('add_query1.html',msg=msg,sid=sid,act=act,data=data,data2=data2)


@app.route('/admin2', methods=['GET', 'POST'])
def admin2():
    msg=""
    mycursor = mydb.cursor()
    if request.method=='POST':
        input1=request.form['input']
        output=request.form['output']
        link=request.form['link']

        if link is None or link=="":
            url=""
        else:
            url=' <a href='+link+' target="_blank">Click Here</a>'

        output+=url
        
        mycursor.execute("SELECT max(id)+1 FROM cc_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        sql = "INSERT INTO cc_data(id,input,output) VALUES (%s,%s,%s)"
        val = (maxid,input1,output)
        mycursor.execute(sql, val)
        mydb.commit()

        
        print(mycursor.rowcount, "Added Success")
        
        return redirect(url_for('view_data',msg='success'))

    return render_template('admin2.html',msg=msg)

@app.route('/view_user', methods=['GET', 'POST'])
def view_user():
    value=[]
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM cc_register")
    data = mycursor.fetchall()

    
    return render_template('view_user.html', data=data)

@app.route('/page', methods=['GET', 'POST'])
def page():
    fn=request.args.get("fn")
   
    
    return render_template('page.html',fn=fn)



@app.route('/register',methods=['POST','GET'])
def register():
    msg=""
    act=""
    mycursor = mydb.cursor()
    name=""
    mobile=""
    mess=""
    uid=""
    if request.method=='POST':
        
        uname=request.form['uname']
        name=request.form['name']     
        mobile=request.form['mobile']
        email=request.form['email']
        location=request.form['location']
        pass1=request.form['pass']

        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        mycursor.execute("SELECT count(*) FROM cc_register where uname=%s",(uname, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM cc_register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            
            uid=str(maxid)
            sql = "INSERT INTO cc_register(id, name, mobile, email, location,uname, pass,otp,status) VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s)"
            val = (maxid, name, mobile, email, location, uname, pass1,'','0')
            msg="success"
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "record inserted.")
           
        else:
            msg="fail"
            
    return render_template('register.html',msg=msg,mobile=mobile,name=name,mess=mess,uid=uid)


def lg_translate(lg,output):
    result=""
    recognized_text=output
    recognizer = sr.Recognizer()
    translator = Translator()
    try:
        available_languages = {
            'ta': 'Tamil',
            'hi': 'Hindi',
            'ml': 'Malayalam',
            'kn': 'Kannada',
            'te': 'Telugu',
            'mr': 'Marathi',
            'ur': 'Urdu',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'fr': 'French'
        }

        print("Available languages:")
        for code, language in available_languages.items():
            print(f"{code}: {language}")

        #selected_languages = input("Enter the language codes (comma-separated) you want to translate to: ").split(',')
        selected_languages=lg.split(',')
       
        for lang_code in selected_languages:
            lang_code = lang_code.strip()
            if lang_code in available_languages:
                translated = translator.translate(recognized_text, dest=lang_code)
                print(f"Translation in {available_languages[lang_code]} ({lang_code}): {translated.text}")

                result=translated.text
               

            else:
                print(f"Language code {lang_code} not available.")

        
    except Exception as e:
        print("An error occurred during translation:", e)

    return result
    ###

####
def translate_text(text, source_language, target_language):
    api_key = 'AIzaSyDW9tvaQUsywmaILt73Go8Fy5mU6ILOixU'  # Replace with your API key
    url = f'https://translation.googleapis.com/language/translate/v2?key={api_key}'
    payload = {
        'q': text,
        'source': source_language,
        'target': target_language,
        'format': 'text'
    }
    response = requests.post(url, json=payload)
    translation_data = response.json()
    translated_text = translation_data
    #translation_data['data']['translations'][0]['translatedText']
    return translated_text

            
@app.route('/bot', methods=['GET', 'POST'])
def bot():
    msg=""
    output=""
    uname=""
    mm=""
    s=""
    xn=0
    val=""
    ##
    result=""
    
    
    ##
    if 'username' in session:
        uname = session['username']
    mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="",
      charset="utf8",
      database="lawyerbot"
    )

    cnt=0
   
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM cc_register where uname=%s",(uname, ))
    value = mycursor.fetchone()
    
    mycursor.execute("SELECT * FROM cc_data order by rand() limit 0,10")
    data=mycursor.fetchall()
            
    if request.method=='POST':
        msg_input=request.form['msg_input']
        lg=request.form['language']
        
        text=msg_input

        ff=open("static/det.txt","r")
        qry_st=ff.read()
        ff.close()
        ##
        #NLP-Preprocessing
        #nlp=STOPWORDS
        #def remove_stopwords(text):
        #    clean_text=' '.join([word for word in text.split() if word not in nlp])
        #    return clean_text
        ##
        #txt=remove_stopwords(msg_input)
        ##
        stemmer = PorterStemmer()
    
        from wordcloud import STOPWORDS
        STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 
                          'im', 'll', 'y', 've', 'u', 'ur', 'don', 
                          'p', 't', 's', 'aren', 'kp', 'o', 'kat', 
                          'de', 're', 'amp', 'will'])

        def lower(text):
            return text.lower()

        def remove_specChar(text):
            return re.sub("#[A-Za-z0-9_]+", ' ', text)

        def remove_link(text):
            return re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', text)

        def remove_stopwords(text):
            return " ".join([word for word in 
                             str(text).split() if word not in STOPWORDS])

        def stemming(text):
            return " ".join([stemmer.stem(word) for word in text.split()])

        #def lemmatizer_words(text):
        #    return " ".join([lematizer.lemmatize(word) for word in text.split()])

        def cleanTxt(text):
            text = lower(text)
            text = remove_specChar(text)
            text = remove_link(text)
            text = remove_stopwords(text)
            text = stemming(text)
            
            return text

        

        #show the clean text
        #dat=df.head()
        #data=[]
        #for ss in dat.values:
        #    data.append(ss)
        #msg_input=data
        mm=""
        mm1=""
        ######################

        

        
        if msg_input=="" or msg_input=="hi":
            s=1
            output="How can i help you?"
            if lg=="":
                val=json.dumps(output)
            else:
                val=lg_translate(lg,output)                
            return val
            #return json.dumps(output)
        else:
            
            if qry_st=="1":
                clean_msg=cleanTxt(msg_input)
                print("uuuu")
                print(clean_msg)
                cleaned='%'+clean_msg+'%'
                print(cleaned)
                mycursor.execute("SELECT count(*) FROM cc_data where user_query like %s or offense like %s or description like %s",(cleaned,cleaned,cleaned))
                cnt1=mycursor.fetchone()[0]
                if cnt1>0:
                    mm='%'+clean_msg+'%'
                else:
                    mm='%'+msg_input+'%'

                ###
                mycursor.execute("SELECT count(*) FROM cc_data where user_query like %s or offense like %s or description like %s",(cleaned,cleaned,cleaned))
                cnt12=mycursor.fetchone()[0]
                if cnt12>0:
                    mm1='%'+clean_msg+'%'
                else:
                    mm1='%'+msg_input+'%'
                ###
                
                mycursor.execute("SELECT count(*) FROM cc_data where user_query like %s or offense like %s or description like %s",(mm,mm,mm))
                cnt=mycursor.fetchone()[0]



                
                if cnt>0:
                    dd3=""
                    sid=0
                    mycursor.execute("SELECT * FROM cc_data where user_query like %s or offense like %s or description like %s limit 0,5",(mm,mm,mm))
                    dd=mycursor.fetchall()
                    for dd1 in dd:
                        sid=dd1[0]
                        
                        dd3+="<br><h5>IPC Section: "+dd1[4]+"</h5><br>Description:<br>"+dd1[1]

                        if dd1[2]=="":
                            s=1
                        else:
                            
                            dd3+="<br><br>Offense:<br>"+dd1[2]


                        if dd1[3]=="":
                            s=1
                        else:
                            dd3+="<br><br>Punishment:<br>"+dd1[3]+"<br>"

                    dff=[]
                    dff2=""
                    
                    mycursor.execute("SELECT count(*) FROM cc_contact")
                    cnt4=mycursor.fetchone()[0]
                    if cnt4>0:
                        mycursor.execute("SELECT distinct(district) FROM cc_contact")
                        dd4=mycursor.fetchall()
                        for dd41 in dd4:
                            dff.append(dd41[0])
                        dff2=",".join(dff)
                        dd3+="<br><br>Which District Lawyer contacts required?<br>"
                        dd3+="("+dff2+")"
                        ff=open("static/det.txt","w")
                        ff.write("2")
                        ff.close()

                    output=dd3
                    
                

                else:
                    ####mm1
                    print("aa")
                    print(mm1)
                    dd3=""
                    sid=0
                    mycursor.execute("SELECT count(*) FROM cc_data where user_query like %s or offense like %s or description like %s",(mm1,mm1,mm1))
                    cnt11=mycursor.fetchone()[0]
                    if cnt11>0:
                                        
                        mycursor.execute("SELECT * FROM cc_data where scheme like %s limit 0,1",(mm1,mm1,mm1))
                        ddx=mycursor.fetchall()
                        for dd1 in ddx:
                            sid=dd1[0]
                   

                        dff=[]
                        dff2=""
                        
                        mycursor.execute("SELECT count(*) FROM cc_contact")
                        cnt4=mycursor.fetchone()[0]
                        if cnt4>0:
                            mycursor.execute("SELECT distinct(district) FROM cc_contact")
                            dd4=mycursor.fetchall()
                            for dd41 in dd4:
                                dff.append(dd41[0])
                            dff2=", ".join(dff)
                            dd3+="<br><br>Which District Lawyer contacts required?<br>"
                            dd3+="("+dff2+")"
                            ff=open("static/det.txt","w")
                            ff.write("2")
                            ff.close()

                                        
                        output=dd3
                    
                    else:                    
                        if msg_input=="":
                            output="How can i help you?"
                        else:
                            output="Sorry, No Results Found!"
                            ff=open("static/det.txt","w")
                            ff.write("1")
                            ff.close()
                if lg=="":
                    val=json.dumps(output)
                else:
                    val=lg_translate(lg,output)                
                return val
                ####################

            

            elif qry_st=="2":
                clean_msg=cleanTxt(msg_input)
                print(clean_msg)
                cleaned='%'+clean_msg+'%'
                
                mycursor.execute("SELECT count(*) FROM cc_contact where district like %s",(cleaned,))
                cnt1=mycursor.fetchone()[0]
                if cnt1>0:
                    mm='%'+clean_msg+'%'
                else:
                    mm='%'+msg_input+'%'
                
                
                mycursor.execute("SELECT count(*) FROM cc_contact where district like %s",(mm,))
                cnt=mycursor.fetchone()[0]


                if cnt>0:
                    dd3=""
                    ff=open("static/section.txt","r")
                    sidd=ff.read()
                    ff.close()
                    mycursor.execute("SELECT * FROM cc_contact where district like %s ",(mm,))
                    dd=mycursor.fetchall()
                    for dd1 in dd:
                        dd3+="<br>Lawyer Name: "+dd1[3]
                        dd3+="<br>Address: "+dd1[6]+", "+dd1[2]
                        dd3+="<br>Mobile No.: "+str(dd1[4])+", Email:"+dd1[5]
                        
                        
                    output=dd3
                    ff=open("static/det.txt","w")
                    ff.write("1")
                    ff.close()
                

                else:
                    if msg_input=="":
                        output="How can i help you?"
                    else:
                        output="Sorry, No Results Found!"
                        ff=open("static/det.txt","w")
                        ff.write("1")
                        ff.close()

                ##################
                if lg=="":
                    val=json.dumps(output)
                else:
                    val=lg_translate(lg,output)                
                return val
                #return json.dumps(output)
                ##################################


    return render_template('bot.html',msg=msg,output=output,uname=uname,data=data,value=value)   


#BERT-Feature Extraction
def BERT():
    super(BERTLM, self).__init__()
    self.vocab = vocab
    self.embed_dim =embed_dim
    self.tok_embed = Embedding(self.vocab.size, embed_dim, self.vocab.padding_idx)
    self.pos_embed = LearnedPositionalEmbedding(embed_dim, device=local_rank)
    self.seg_embed = Embedding(2, embed_dim, None)

    self.out_proj_bias = nn.Parameter(torch.Tensor(self.vocab.size))

    self.layers = nn.ModuleList()
    for i in range(layers):
        self.layers.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout))
    self.emb_layer_norm = LayerNorm(embed_dim)
    self.one_more = nn.Linear(embed_dim, embed_dim)
    self.one_more_layer_norm = LayerNorm(embed_dim)
    self.one_more_nxt_snt = nn.Linear(embed_dim, embed_dim) 
    self.nxt_snt_pred = nn.Linear(embed_dim, 1)
    self.dropout = dropout
    self.device = local_rank

    if approx == "none":
        self.approx = None
    elif approx == "adaptive":
        self.approx = nn.AdaptiveLogSoftmaxWithLoss(self.embed_dim, self.vocab.size, [10000, 20000, 200000])
    else:
        raise NotImplementedError("%s has not been implemented"%approx)
    self.reset_parameters()

def reset_parameters(self):
    nn.init.constant_(self.out_proj_bias, 0.)
    nn.init.constant_(self.nxt_snt_pred.bias, 0.)
    nn.init.constant_(self.one_more.bias, 0.)
    nn.init.constant_(self.one_more_nxt_snt.bias, 0.)
    nn.init.normal_(self.nxt_snt_pred.weight, std=0.02)
    nn.init.normal_(self.one_more.weight, std=0.02)
    nn.init.normal_(self.one_more_nxt_snt.weight, std=0.02)

def work(self, inp, seg=None, layers=None):
   
    if layers is not None:
        tot_layers = len(self.layers)
        for x in layers:
            if not (-tot_layers <= x < tot_layers):
                raise ValueError('layer %d out of range '%x)
        layers = [ (x+tot_layers if x <0 else x) for x in layers]
        max_layer_id = max(layers)
    
    seq_len, bsz = inp.size()
    if seg is None:
        seg = torch.zeros_like(inp)
    x = self.tok_embed(inp) + self.seg_embed(seg) + self.pos_embed(inp)
    x = self.emb_layer_norm(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    padding_mask = torch.eq(inp, self.vocab.padding_idx)
    if not padding_mask.any():
        padding_mask = None
    
    xs = []
    for layer_id, layer in enumerate(self.layers):
        x, _ ,_ = layer(x, self_padding_mask=padding_mask)
        xs.append(x)
        if layers is not None and layer_id >= max_layer_id:
            break
    
    if layers is not None:
        x = torch.stack([xs[i] for i in layers])
        z = torch.tanh(self.one_more_nxt_snt(x[:,0,:,:]))
    else:
        z = torch.tanh(self.one_more_nxt_snt(x[0]))
    return x, z

def forward(self, truth, inp, seg, msk, nxt_snt_flag):
    seq_len, bsz = inp.size()
    x = self.tok_embed(inp) + self.seg_embed(seg) + self.pos_embed(inp)
    x = self.emb_layer_norm(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    padding_mask = torch.eq(truth, self.vocab.padding_idx)
    if not padding_mask.any():
        padding_mask = None
    for layer in self.layers:
        x, _ ,_ = layer(x, self_padding_mask=padding_mask)

    masked_x = x.masked_select(msk.unsqueeze(-1))
    masked_x = masked_x.view(-1, self.embed_dim)
    gold = truth.masked_select(msk)
    
    y = self.one_more_layer_norm(gelu(self.one_more(masked_x)))
    out_proj_weight = self.tok_embed.weight

    if self.approx is None:
        log_probs = torch.log_softmax(F.linear(y, out_proj_weight, self.out_proj_bias), -1)
    else:
        log_probs = self.approx.log_prob(y)

    loss = F.nll_loss(log_probs, gold, reduction='mean')

    z = torch.tanh(self.one_more_nxt_snt(x[0]))
    nxt_snt_pred = torch.sigmoid(self.nxt_snt_pred(z).squeeze(1))
    nxt_snt_acc = torch.eq(torch.gt(nxt_snt_pred, 0.5), nxt_snt_flag).float().sum().item()
    nxt_snt_loss = F.binary_cross_entropy(nxt_snt_pred, nxt_snt_flag.float(), reduction='mean')
    
    tot_loss = loss + nxt_snt_loss
    
    _, pred = log_probs.max(-1)
    tot_tokens = msk.float().sum().item()
    acc = torch.eq(pred, gold).float().sum().item()
    
    return (pred, gold), tot_loss, acc, tot_tokens, nxt_snt_acc, bsz
####
#LSTM-Classification
class LSTM():
    INPUT_VECTOR_LENGTH = 20
    OUTPUT_VECTORLENGTH = 20
    minimum_length = 2
    maximum_length = 20
    sample_size = 30000 
    WORD_START = 1
    WORD_PADDING = 0

    def extract_converstionIDs(conversation_lines):
        conversations = []
        for line in conversation_lines[:-1]:
            split_line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
            conversations.append(split_line.split(','))
        return conversations

    def extract_quesans_pairs(linetoID_mapping,conversations):
        questions = []
        answers = []
        for con in conversations:
            for i in range(len(con)-1):
                questions.append(linetoID_mapping[con[i]])
                answers.append(linetoID_mapping[con[i+1]])
        return questions,answers
    def transform_text(input_text):
        input_text = input_text.lower()
        input_text = re.sub(r"I'm", "I am", input_text)
        input_text = re.sub(r"he's", "he is", input_text)
        input_text = re.sub(r"she's", "she is", input_text)
        input_text = re.sub(r"it's", "it is", input_text)
        input_text = re.sub(r"that's", "that is", input_text)
        input_text = re.sub(r"what's", "that is", input_text)
        input_text = re.sub(r"where's", "where is", input_text)
        input_text = re.sub(r"how's", "how is", input_text)
        input_text = re.sub(r"\'ll", " will", input_text)
        input_text = re.sub(r"\'ve", " have", input_text)
        input_text = re.sub(r"\'re", " are", input_text)
        input_text = re.sub(r"\'d", " would", input_text)
        input_text = re.sub(r"\'re", " are", input_text)
        input_text = re.sub(r"won't", "will not", input_text)
        input_text = re.sub(r"can't", "cannot", input_text)
        input_text = re.sub(r"n't", " not", input_text)
        input_text = re.sub(r"'til", "until", input_text)
        input_text = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", input_text)
        input_text = " ".join(input_text.split())
        return input_text

    def filter_ques_ans(clean_questions,clean_answers):
        # Filter out the questions that are too short/long
        short_questions_temp = []
        short_answers_temp = []
        for i, question in enumerate(clean_questions):
            if len(question.split()) >= minimum_length and len(question.split()) <= maximum_length:
                short_questions_temp.append(question)
                short_answers_temp.append(clean_answers[i])
        short_questions = []
        short_answers = []
        for i, answer in enumerate(short_answers_temp):
            if len(answer.split()) >= minimum_length and len(answer.split()) <= maximum_length:
                short_answers.append(answer)
                short_questions.append(short_questions_temp[i])
        return short_questions,short_answers

    def create_vocabulary(tokenized_ques,tokenized_ans):
        vocabulary = {}
        for question in tokenized_ques:
            for word in question:
                if word not in vocabulary:
                    vocabulary[word] = 1
                else:
                    vocabulary[word] += 1
        for answer in tokenized_ans:
            for word in answer:
                if word not in vocabulary:
                    vocabulary[word] = 1
                else:
                    vocabulary[word] += 1  
        return vocabulary

    def create_encoding_decoding(vocabulary):
        threshold = 15
        count = 0
        for k,v in vocabulary.items():
            if v >= threshold:
                count += 1
        vocab_size  = 2 
        encoding = {}
        decoding = {1: 'START'}
        for word, count in vocabulary.items():
            if count >= threshold:
                encoding[word] = vocab_size 
                decoding[vocab_size ] = word
                vocab_size += 1
        return encoding,decoding,vocab_size
    def transform(encoding, data, vector_size=20):
        transformed_data = np.zeros(shape=(len(data), vector_size))
        for i in range(len(data)):
            for j in range(min(len(data[i]), vector_size)):
                try:
                    transformed_data[i][j] = encoding[data[i][j]]
                except:
                    transformed_data[i][j] = encoding['<UNKNOWN>']
        return transformed_data
    def create_gloveEmbeddings(encoding,size):
        file = open(GLOVE_MODEL, mode='rt', encoding='utf8')
        words = set()
        word_to_vec_map = {}
        for line in file:
            line = line.strip().split()
            word = line[0]
            words.add(word)
            word_to_vec_map[word] = np.array(line[1:], dtype=np.float64)
        embedding_matrix = np.zeros((size, 50))
        for word,index in encoding.items():
            try:
                embedding_matrix[index, :] = word_to_vec_map[word.lower()]
            except: continue
        return embedding_matrix

    def create_model(dict_size,embed_layer,hidden_dim):
    
        encoder_inputs = Input(shape=(maximum_length, ), dtype='int32',)
        encoder_embedding = embed_layer(encoder_inputs)
        encoder_LSTM = LSTM(hidden_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
        decoder_inputs = Input(shape=(maximum_length, ), dtype='int32',)
        decoder_embedding = embed_layer(decoder_inputs)
        decoder_LSTM = LSTM(hidden_dim, return_state=True, return_sequences=True)
        decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
        outputs = TimeDistributed(Dense(dict_size, activation='softmax'))(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], outputs)
        return model

    def prediction_answer(user_input,model):
        transformed_input = transform_text(user_input)
        input_tokens = [nltk.word_tokenize(transformed_input)]
        input_tokens = [input_tokens[0][::-1]]  #reverseing input seq
        encoder_input = transform(encoding, input_tokens, 20)
        decoder_input = np.zeros(shape=(len(encoder_input), OUTPUT_VECTORLENGTH))
        decoder_input[:,0] = WORD_START
        for i in range(1, OUTPUT_VECTORLENGTH):
            pred_output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
            decoder_input[:,i] = pred_output[:,i]
        return pred_output

##########################
@app.route('/sign')
def sign():
    return render_template('sign.html')

@app.route('/signUpUser', methods=['POST'])
def signUpUser():
    user =  request.form['username'];
    password = request.form['password'];

    print(password)
    return json.dumps({'status':'OK','user':user,'pass':password});


@app.route('/view_data', methods=['GET', 'POST'])
def view_data():
    data=[]
    msg=request.args.get("msg")
    act=request.args.get("act")
    url=""
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM cc_data")
    data1 = mycursor.fetchall()

    for dat in data1:
        dt=[]
        txt=dat[1]
        t=txt.replace("\t\r\n","<br>")
        #if "\t\r\n" in dat[2]:

        dt.append(dat[0])
        dt.append(t)
        dt.append(dat[2])
        dt.append(dat[3])
        dt.append(dat[4])
        
        data.append(dt)
            
            
    

    
    
    
    return render_template('view_data.html',msg=msg,act=act,data=data)

@app.route('/down', methods=['GET', 'POST'])
def down():
    fn = request.args.get('fname')
    path="static/upload/"+fn
    return send_file(path, as_attachment=True)

@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
