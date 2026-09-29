import sqlite3
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE=Path(__file__).resolve().parent
DATA=BASE/"data"/"loan_dataset.csv"; DB=BASE/"data"/"easyloan.db"
FEATURES=["age","income","loan_amount","loan_term_months","credit_score","current_debt","current_loans","employment","credit_history"]

df=pd.read_csv(DATA); X=df[FEATURES]; y=df["approved"]
num=["age","income","loan_amount","loan_term_months","credit_score","current_debt","current_loans"]
cat=["employment","credit_history"]
prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat)],remainder="passthrough")
model=Pipeline([("prep",prep),("rf",RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced"))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model.fit(Xtr,ytr); yp=model.predict(Xte)
metrics={k:v for k,v in [("Accuracy",accuracy_score(yte,yp)),("Precision",precision_score(yte,yp,zero_division=0)),("Recall",recall_score(yte,yp,zero_division=0)),("F1-score",f1_score(yte,yp,zero_division=0))]}

def conn(): return sqlite3.connect(DB)
with conn() as c:
    c.execute("""CREATE TABLE IF NOT EXISTS applications(
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, phone TEXT, age INTEGER,
    income REAL, loan REAL, term INTEGER, credit INTEGER, debt REAL, loans INTEGER,
    employment TEXT, history TEXT, prediction INTEGER, probability REAL,
    monthly REAL, created TEXT DEFAULT CURRENT_TIMESTAMP)""")

def money(x): return f"{x:,.0f}".replace(",",".")

def parse_money(s): return float(s.replace(".","").replace(",","").strip())

def payment(p, annual, n):
    r=annual/12
    return p/n if r==0 else p*r*(1+r)**n/((1+r)**n-1)

class App:
    def __init__(self,r):
        self.r=r; r.title("AI - Ứng dụng vay"); r.geometry("950x700")
        h=tk.Frame(r,bg="#173F5F"); h.pack(fill="x")
        tk.Label(h,text="AI",bg="#173F5F",fg="white",font=("Segoe UI",24,"bold")).pack(anchor="w",padx=25,pady=(15,0))
        tk.Label(h,text="Đăng ký khoản vay • Phân tích hồ sơ bằng Machine Learning",bg="#173F5F",fg="white").pack(anchor="w",padx=28,pady=(0,15))
        n=ttk.Frame(r,padding=10); n.pack(fill="x")
        ttk.Button(n,text="Hồ sơ vay",command=self.form).pack(side="left",padx=4)
        ttk.Button(n,text="Lịch sử",command=self.history).pack(side="left",padx=4)
        ttk.Button(n,text="Đánh giá mô hình",command=self.metrics).pack(side="left",padx=4)
        self.body=ttk.Frame(r,padding=25); self.body.pack(fill="both",expand=True); self.form()

    def clear(self):
        for w in self.body.winfo_children(): w.destroy()

    def form(self):
        self.clear(); ttk.Label(self.body,text="ĐĂNG KÝ KHOẢN VAY",font=("Segoe UI",20,"bold")).grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,15))
        self.v={}
        fields=[("Họ và tên","name",""),("Số điện thoại","phone",""),("Tuổi","age","25"),("Thu nhập/tháng (VNĐ)","income","20000000"),("Số tiền muốn vay (VNĐ)","loan","100000000"),("Tổng nợ hiện tại (VNĐ)","debt","20000000"),("Điểm tín dụng","credit","720")]
        for i,(lab,key,val) in enumerate(fields,1):
            ttk.Label(self.body,text=lab).grid(row=i,column=0,sticky="w",pady=6)
            self.v[key]=tk.StringVar(value=val); ttk.Entry(self.body,textvariable=self.v[key],width=40).grid(row=i,column=1,padx=18,sticky="ew",pady=6)
        combos=[("Thời hạn vay","term",[6,12,18,24,36,48,60],"24"),("Số khoản vay hiện tại","loans",[0,1,2,3,4,5],"1"),("Nghề nghiệp","employment",["Ổn định","Hợp đồng","Tự doanh","Không ổn định"],"Ổn định"),("Lịch sử tín dụng","history",["Tốt","Trung bình","Xấu"],"Tốt")]
        for i,(lab,key,vals,default) in enumerate(combos,8):
            ttk.Label(self.body,text=lab).grid(row=i,column=0,sticky="w",pady=6)
            self.v[key]=tk.StringVar(value=default); ttk.Combobox(self.body,textvariable=self.v[key],values=vals,state="readonly",width=37).grid(row=i,column=1,padx=18,sticky="w",pady=6)
        self.body.columnconfigure(1,weight=1)
        ttk.Button(self.body,text="KIỂM TRA KHẢ NĂNG PHÊ DUYỆT",command=self.predict).grid(row=12,column=0,columnspan=2,pady=20,ipadx=20,ipady=8)

    def predict(self):
        try:
            v=self.v; name=v["name"].get().strip()
            if not name: raise ValueError("Vui lòng nhập họ và tên.")
            age=int(v["age"].get()); income=parse_money(v["income"].get()); loan=parse_money(v["loan"].get()); debt=parse_money(v["debt"].get()); credit=int(v["credit"].get())
            term=int(v["term"].get()); loans=int(v["loans"].get())
            if not 18<=age<=100: raise ValueError("Tuổi phải từ 18 đến 100.")
            if income<=0 or loan<=0 or debt<0: raise ValueError("Thu nhập/khoản vay phải > 0 và nợ không âm.")
            if not 300<=credit<=850: raise ValueError("Điểm tín dụng phải từ 300 đến 850.")
            one=pd.DataFrame([{"age":age,"income":income,"loan_amount":loan,"loan_term_months":term,"credit_score":credit,"current_debt":debt,"current_loans":loans,"employment":v["employment"].get(),"credit_history":v["history"].get()}])
            pred=int(model.predict(one)[0]); probs=model.predict_proba(one)[0]; prob=float(probs[pred]); monthly=payment(loan,.12,term)
            with conn() as c:
                c.execute("INSERT INTO applications(name,phone,age,income,loan,term,credit,debt,loans,employment,history,prediction,probability,monthly) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(name,v["phone"].get(),age,income,loan,term,credit,debt,loans,v["employment"].get(),v["history"].get(),pred,prob,monthly))
            self.result(pred,prob,loan,term,monthly)
        except Exception as e: messagebox.showerror("Lỗi dữ liệu",str(e))

    def result(self,p,prob,loan,term,monthly):
        self.clear(); ttk.Label(self.body,text="✓" if p else "!",font=("Segoe UI",42,"bold")).pack(pady=(20,0))
        ttk.Label(self.body,text="CÓ KHẢ NĂNG ĐƯỢC PHÊ DUYỆT" if p else "CHƯA CÓ KHẢ NĂNG ĐƯỢC PHÊ DUYỆT",font=("Segoe UI",22,"bold")).pack(pady=5)
        ttk.Label(self.body,text=f"Xác suất theo mô hình: {prob*100:.2f}%",font=("Segoe UI",14)).pack(pady=8)
        box=ttk.LabelFrame(self.body,text="Thông tin khoản vay",padding=20); box.pack(fill="x",padx=100,pady=20)
        for i,(a,b) in enumerate([("Số tiền vay",money(loan)+" VNĐ"),("Thời hạn",f"{term} tháng"),("Dự kiến trả/tháng",money(monthly)+" VNĐ"),("Lãi suất minh họa","12%/năm")]):
            ttk.Label(box,text=a).grid(row=i,column=0,sticky="w",pady=6); ttk.Label(box,text=b,font=("Segoe UI",10,"bold")).grid(row=i,column=1,sticky="e",pady=6)
        ttk.Button(self.body,text="ĐĂNG KÝ HỒ SƠ KHÁC",command=self.form).pack(pady=6)
        ttk.Button(self.body,text="XEM LỊCH SỬ",command=self.history).pack(pady=6)

    def history(self):
        self.clear(); ttk.Label(self.body,text="LỊCH SỬ HỒ SƠ VAY",font=("Segoe UI",20,"bold")).pack(anchor="w",pady=(0,15))
        cols=("id","name","loan","term","result","prob","date"); t=ttk.Treeview(self.body,columns=cols,show="headings")
        for c,h in zip(cols,["ID","Khách hàng","Khoản vay","Thời hạn","Kết quả","Xác suất","Ngày"]): t.heading(c,text=h); t.column(c,width=120,anchor="center")
        t.pack(fill="both",expand=True)
        with conn() as c: rows=c.execute("SELECT id,name,loan,term,prediction,probability,created FROM applications ORDER BY id DESC").fetchall()
        for x in rows: t.insert("", "end",values=(x[0],x[1],money(x[2])+" VNĐ",f"{x[3]} tháng","Có khả năng" if x[4] else "Chưa có khả năng",f"{x[5]*100:.1f}%",x[6]))

    def metrics(self):
        self.clear(); ttk.Label(self.body,text="ĐÁNH GIÁ MÔ HÌNH",font=("Segoe UI",20,"bold")).pack(anchor="w",pady=(0,20))
        b=ttk.LabelFrame(self.body,text="Random Forest Classifier",padding=25); b.pack(fill="x",padx=100)
        for k,val in metrics.items(): ttk.Label(b,text=k,font=("Segoe UI",12)).pack(anchor="w",pady=5); ttk.Label(b,text=f"{val*100:.2f}%",font=("Segoe UI",15,"bold")).pack(anchor="w")
        ttk.Label(self.body,text="Dataset tổng hợp phục vụ mục đích học tập.").pack(pady=20)

root=tk.Tk(); App(root); root.mainloop()
