import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
BASE=os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(BASE,"data"),exist_ok=True); os.makedirs(os.path.join(BASE,"results"),exist_ok=True)
rng=np.random.default_rng(42); n=2000
dept=rng.choice(["Sales","Engineering","Operations","HR","Finance","Support"],n)
tenure=rng.integers(0,20,n); salary=rng.normal(42000,12000,n).clip(22000,90000); satisfaction=rng.integers(1,6,n); overtime=rng.binomial(1,0.3,n)
logit=0.5 -0.06*tenure -0.20*satisfaction +0.7*overtime +0.4*(dept=="Sales") -0.00002*salary
prob=1/(1+np.exp(-logit)); attr=rng.binomial(1,prob)
df=pd.DataFrame({"employee_id":range(1,n+1),"department":dept,"tenure_years":tenure,"salary_gbp":salary.round(2),"job_satisfaction":satisfaction,"overtime":overtime,"attrition":attr})
df["tenure_band"]=pd.cut(df.tenure_years,bins=[-1,2,5,10,100],labels=["0-2","3-5","6-10","10+"])
df.to_csv(os.path.join(BASE,"data","workforce_data_synthetic.csv"),index=False)
by_dept=df.groupby("department")["attrition"].mean().sort_values(ascending=False)
by_band=df.groupby("tenure_band",observed=True)["attrition"].mean()
X=pd.get_dummies(df[["tenure_years","salary_gbp","job_satisfaction","overtime","department"]],drop_first=True); y=df.attrition
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
sc=StandardScaler(); m=LogisticRegression(max_iter=1000); m.fit(sc.fit_transform(Xtr),ytr); pred=m.predict(sc.transform(Xte)); auc=roc_auc_score(yte,m.predict_proba(sc.transform(Xte))[:,1]); acc=accuracy_score(yte,pred)
with open(os.path.join(BASE,"results","results_summary.txt"),"w") as f:
    f.write("HR Workforce Attrition Analytics — SYNTHETIC RESULTS\n")
    f.write(f"Employees: {n}, Overall attrition rate: {y.mean():.4f}\n")
    f.write("Attrition by department:\n")
    for k,v in by_dept.items(): f.write(f"  {k}: {v:.4f}\n")
    f.write("Attrition by tenure band:\n")
    for k,v in by_band.items(): f.write(f"  {k}: {v:.4f}\n")
    f.write(f"Logistic Regression — accuracy={acc:.4f} ROC-AUC={auc:.4f}\n")
print(f"attrition={y.mean():.4f} acc={acc:.4f} auc={auc:.4f} top_dept={by_dept.index[0]} {by_dept.iloc[0]:.4f}")
