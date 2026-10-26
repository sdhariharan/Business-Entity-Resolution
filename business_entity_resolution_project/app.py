from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib, os, re, zipfile
from rapidfuzz import fuzz, process

app = Flask(__name__)
MODEL_PATH = "model/business_entity_resolution_model.joblib"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_FEATURES = [
    "name_ratio","name_partial_ratio","name_token_sort","name_token_set",
    "address_ratio","address_partial_ratio","address_token_sort","address_token_set",
    "name_jaccard","address_jaccard","number_match","country_match",
    "name_length_1","name_length_2","address_length_1","address_length_2"
]

model = None
BEST_THRESHOLD = 0.80
FEATURE_NAMES = DEFAULT_FEATURES

try:
    if os.path.exists(MODEL_PATH):
        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict) and "model" in loaded:
            model = loaded["model"]
            BEST_THRESHOLD = float(loaded.get("threshold", 0.80))
            FEATURE_NAMES = loaded.get("features") or DEFAULT_FEATURES
        else:
            model = loaded
        print("Model loaded:", type(model))
    else:
        print("Model not found. Put it at:", MODEL_PATH)
except Exception as exc:
    print(f"Could not load model at {MODEL_PATH}: {exc}")
    print("Install the pinned project dependencies from requirements.txt or retrain the model with the current environment.")

def normalize_text(value):
    if pd.isna(value): return ""
    text = str(value).lower().replace("&", " and ")
    replacements = {
        "corporation":"corp","company":"co","limited":"ltd","private":"pvt",
        "incorporated":"inc","road":"rd","street":"st","avenue":"ave",
        "boulevard":"blvd","apartment":"apt"
    }
    for old,new in replacements.items():
        text = re.sub(r"\b"+re.escape(old)+r"\b", new, text)
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ",text)).strip()

def nums(text):
    return set(re.findall(r"\d+", str(text)))

def create_features(a,b):
    n1,n2=a["name_norm"],b["name_norm"]
    ad1,ad2=a["address_norm"],b["address_norm"]
    nt1,nt2=set(n1.split()),set(n2.split())
    at1,at2=set(ad1.split()),set(ad2.split())
    nu,au=nt1|nt2,at1|at2
    nj=len(nt1&nt2)/len(nu) if nu else 0
    aj=len(at1&at2)/len(au) if au else 0
    x,y=nums(ad1),nums(ad2)
    u=x|y
    nm=len(x&y)/len(u) if u else 0
    return [
        fuzz.ratio(n1,n2)/100, fuzz.partial_ratio(n1,n2)/100,
        fuzz.token_sort_ratio(n1,n2)/100, fuzz.token_set_ratio(n1,n2)/100,
        fuzz.ratio(ad1,ad2)/100, fuzz.partial_ratio(ad1,ad2)/100,
        fuzz.token_sort_ratio(ad1,ad2)/100, fuzz.token_set_ratio(ad1,ad2)/100,
        nj, aj, nm, int(a["country_norm"]==b["country_norm"]),
        len(n1),len(n2),len(ad1),len(ad2)
    ]

def prepare(df):
    required=["entity_id","business_name","business_address","country"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError("Missing columns: "+", ".join(missing))
    df=df.copy().fillna("")
    df["name_norm"]=df["business_name"].apply(normalize_text)
    df["address_norm"]=df["business_address"].apply(normalize_text)
    df["country_norm"]=df["country"].apply(normalize_text)
    return df

def candidates(row,names,addresses,limit=15):
    a=process.extract(row["name_norm"],names,scorer=fuzz.token_set_ratio,limit=limit)
    b=process.extract(row["address_norm"],addresses,scorer=fuzz.token_set_ratio,limit=limit)
    return list({r[2] for r in a+b})

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({"status":"running","model_loaded":model is not None,
                    "threshold":BEST_THRESHOLD,"feature_count":len(FEATURE_NAMES)})

@app.route("/api/match", methods=["POST"])
def match():
    try:
        if model is None:
            return jsonify({"success":False,"error":
                "Model missing. Put your .joblib file at model/business_entity_resolution_model.joblib"}),500
        for k in ("source1","source2","source3"):
            if k not in request.files:
                return jsonify({"error":f"{k} file is required"}),400
        s1=prepare(pd.read_csv(request.files["source1"],sep="\t",dtype=str))
        s2=prepare(pd.read_csv(request.files["source2"],sep="\t",dtype=str))
        s3=prepare(pd.read_csv(request.files["source3"],sep="\t",dtype=str))
        target=pd.concat([s2,s3],ignore_index=True)
        lookup={r["entity_id"]:r for _,r in target.iterrows()}
        names=dict(zip(target["entity_id"],target["name_norm"]))
        addresses=dict(zip(target["entity_id"],target["address_norm"]))
        matches=[]; candidate_rows=[]; total=0

        for _,a in s1.iterrows():
            cands=candidates(a,names,addresses)
            candidate_rows.append({"source1_entity_id":a["entity_id"],
                                   "candidate_entity_ids":",".join(sorted(cands))})
            feats=[]; ids=[]
            for cid in cands:
                b=lookup.get(cid)
                if b is not None:
                    feats.append(create_features(a,b)); ids.append(cid)
            matched=[]
            if feats:
                X=pd.DataFrame(feats,columns=FEATURE_NAMES)
                probs=model.predict_proba(X)[:,1]
                matched=[cid for cid,p in zip(ids,probs) if p>=BEST_THRESHOLD]
            matched=sorted(set(matched)); total+=len(matched)
            matches.append({"source1_entity_id":a["entity_id"],
                            "matched_entity_ids":",".join(matched)})

        pd.DataFrame(matches).to_csv(OUTPUT_DIR+"/matching_results.tsv",sep="\t",index=False)
        pd.DataFrame(candidate_rows).to_csv(OUTPUT_DIR+"/candidate_pairs.tsv",sep="\t",index=False)

        return jsonify({"success":True,"source1_records":len(s1),"source2_records":len(s2),
                        "source3_records":len(s3),"total_matches":total,
                        "threshold":BEST_THRESHOLD,"preview":matches[:50]})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}),500

@app.route("/api/download/<kind>")
def download(kind):
    files={"matching":"matching_results.tsv","candidates":"candidate_pairs.tsv"}
    if kind=="submission":
        a=OUTPUT_DIR+"/matching_results.tsv"; b=OUTPUT_DIR+"/candidate_pairs.tsv"
        if not (os.path.exists(a) and os.path.exists(b)):
            return jsonify({"error":"Run matching first."}),404
        zpath=OUTPUT_DIR+"/business_entity_resolution_submission.zip"
        with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
            z.write(a,"matching_results.tsv"); z.write(b,"candidate_pairs.tsv")
        return send_file(zpath,as_attachment=True,download_name="business_entity_resolution_submission.zip")
    if kind not in files or not os.path.exists(OUTPUT_DIR+"/"+files[kind]):
        return jsonify({"error":"Run matching first."}),404
    return send_file(OUTPUT_DIR+"/"+files[kind],as_attachment=True,download_name=files[kind])

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
