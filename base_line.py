import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

RADIUS=1
POINTS=8
METHOD="uniform"
N_BINS=POINTS*(POINTS-1)+3

def extract_features(image_path):

    img=cv2.imread(image_path,0)

    lbp=local_binary_pattern(img,POINTS,RADIUS,METHOD)

    lbp_hist,_=np.histogram(
        lbp.ravel(),
        bins=N_BINS,
        range=(0,N_BINS)
    )

    lbp_hist=lbp_hist.astype("float")
    lbp_hist/=lbp_hist.sum()+1e-6

    glcm=graycomatrix(
        img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    props=["contrast","dissimilarity","homogeneity","energy"]

    glcm_features=[graycoprops(glcm,p)[0,0] for p in props]

    features=np.hstack([lbp_hist,glcm_features])

    return features


def load_dataset(original_dir,tampered_dir):

    X=[]
    y=[]

    for fname in os.listdir(original_dir):

        feat=extract_features(os.path.join(original_dir,fname))

        X.append(feat)
        y.append(0)

    for fname in os.listdir(tampered_dir):

        feat=extract_features(os.path.join(tampered_dir,fname))

        X.append(feat)
        y.append(1)

    return np.array(X),np.array(y)


if __name__=="__main__":

    X,y=load_dataset("dataset-copy/original","dataset-copy/tampered")

    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    svm=SVC(kernel="rbf",C=1.0,gamma="scale")

    svm.fit(X_train,y_train)

    y_pred=svm.predict(X_test)

    print("Accuracy:",accuracy_score(y_test,y_pred))

    print(classification_report(y_test,y_pred))