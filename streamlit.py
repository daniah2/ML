import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import joblib

from skimage.feature import local_binary_pattern, hog

# -------------------------
# Paths
# -------------------------
data_dir = "/Users/dani/Desktop/machine-learning-project-ml-team-3/MS"

folders = {
    "Healthy": ["Control Axial_crop", "Control Saggital_crop"],
    "MS": ["MS Axial_crop", "MS Saggital_crop"]
}

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="MS Detection Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Medical Color Palette
# -------------------------
medical_palette = [
    "#4A6FA5",  # Blue
    "#6B8E23",  # Olive Green
    "#9370DB",  # Purple
    "#20B2AA",  # Light Sea Green
    "#FF6347",  # Tomato Red
    "#4682B4",  # Steel Blue
    "#32CD32",  # Lime Green
    "#BA55D3",  # Medium Orchid
]

# -------------------------
# Custom CSS
# -------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp {{
    background: linear-gradient(135deg, #F0F8FF 0%, #E6E6FA 50%, #F0FFF0 100%);
    font-family: 'Inter', sans-serif;
    color: #333333;
}}
.main-header {{
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #4A6FA5 0%, #6B8E23 100%);
    color: #ffffff;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 2rem;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(74, 111, 165, 0.3);
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}}
.stTabs > div:first-child {{
    justify-content: center;
}}
.stTabs button {{
    background: linear-gradient(135deg, #4A6FA5 0%, #6B8E23 100%);
    color: #FFFFFF !important;
    font-weight: 600;
    border-radius: 12px 12px 0 0;
    margin: 0 5px;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
}}
.stTabs button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(74, 111, 165, 0.3);
}}
.stTabs button[aria-selected="true"] {{
    background: linear-gradient(135deg, #9370DB 0%, #20B2AA 100%);
    color: #FFFFFF !important;
    font-weight: 700;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Logo + Main Title in one row
# -------------------------
logo_path = "logo.png"  # ضع هنا مسار شعارك
col1, col2 = st.columns([1, 10])  # العمود الأول أصغر للوجو، الثاني أكبر للعنوان

with col1:
    st.image(logo_path, width=150)  # حجم اللوقو

with col2:
    st.markdown('<h1 class="main-header">Multiple Sclerosis Detection from Brain MRI</h1>', unsafe_allow_html=True) 
# -------------------------
# Load Images & Split
# -------------------------
X = []
y = []

for label, subfolders in folders.items():
    class_label = 0 if label=="Healthy" else 1
    for subfolder in subfolders:
        folder_path = os.path.join(data_dir, subfolder)
        for file in os.listdir(folder_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (224,224))
                X.append(img)
                y.append(class_label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["About", "EDA", "Prediction", "Team"])

# -------------------------
# TAB 1: About
# -------------------------
with tab1:
    st.markdown('<p class="section-header"> MS Detection Overview</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    # Problem
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-title">Problem</p>
            <p class="metric-subtitle">
            Multiple Sclerosis (MS) is a chronic neurological disease that damages the 
            white matter in the brain and spinal cord.  
            Diagnosing MS early is difficult and usually requires analyzing MRI scans, 
            which is time-consuming for doctors and may lead to delayed treatment.  
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Solution 
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-title">Solution</p>
            <p class="metric-subtitle">
            We developed a system that uses MRI scans to automatically distinguish 
            between MS patients and healthy individuals.  
            This tool aims to support doctors and researchers by making the detection 
            process faster, clearer, and more consistent.  
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Workflow
    with col3:
        st.markdown(f"""
        <div class="metric-card">
           <p class="metric-title">Our Dataset & Workflow</p>
        <p class="metric-subtitle">
        <b>Dataset:</b><br>
        • Collected from a public MRI dataset for Multiple Sclerosis detection.<br>
        • Contains brain MRI scans of both <b>healthy individuals</b> and <b>MS patients</b>.<br>
        • Each scan was labeled (MS / Healthy) to be used in training the model.<br>

        <br><b>Model:</b><br>
        • We applied feature-extraction methods:<br>
        &nbsp;&nbsp;- <b>LBP (Local Binary Pattern):</b> to capture texture patterns in MRI scans.<br>
        &nbsp;&nbsp;- <b>HOG (Histogram of Oriented Gradients):</b> to capture shape and edge information.<br>
        • Features were scaled and reduced in size using <b>PCA</b> for efficiency.<br>
        • Final classification was performed with <b>Support Vector Machine (SVM)</b>, chosen after comparing different models for best accuracy and reliability.<br>

      <br><b>Workflow (what we actually did):</b><br>
        <ol style="margin-left:16px;">
          <li>Load the labeled MRI dataset (Healthy vs MS; Axial & Sagittal).</li>
          <li>Preprocess images (resize to 224×224, convert to grayscale).</li>
          <li>Extract features: LBP + HOG, then concatenate.</li>
          <li>Scale features and apply PCA for dimensionality reduction.</li>
          <li><b>Split</b> data (train/test) with stratification (80/20).</li>
          <li>Train a baseline SVM </b> on the training set and evaluate it (accuracy, report, ROC-AUC).</li>
          <li><b>Hyperparameter tuning</b> using RandomizedSearchCV with <b>5-fold Cross-Validation</b>.</li>
          <li>Retrain the <b>best SVM model</b> found by CV on the training data.</li>
          <li>Evaluate the <b>optimized SVM</b> on the test set as the final chosen model.</li>
        </ol>
        </p>
    </div>
    """, unsafe_allow_html=True)
# -------------------------
# TAB 2: EDA + Metrics
# -------------------------
with tab2:
    st.markdown('<p class="section-header">Exploratory Data Analysis</p>', unsafe_allow_html=True)

    # Pie Chart
    unique, counts = np.unique(y, return_counts=True)
    df_counts = pd.DataFrame({'Class':['Healthy','MS'], 'Count':counts})
    
    st.subheader("Class Distribution")
    fig1, ax1 = plt.subplots()
    ax1.pie(df_counts['Count'], labels=df_counts['Class'], autopct='%1.1f%%', colors=[medical_palette[1], medical_palette[0]])
    st.pyplot(fig1)

    # Count Plot
    st.subheader("Class Count")
    fig2, ax2 = plt.subplots()
    sns.countplot(x=y, palette=[medical_palette[1], medical_palette[0]], ax=ax2)
    ax2.set_title("Class Distribution (0=Healthy, 1=MS)")
    counts = np.bincount(y)
    for i, count in enumerate(counts):
        ax2.text(i, count + 2, str(count), ha='center', fontsize=12)
    st.pyplot(fig2)

    # Sample MRI Images
    st.subheader("Sample MRI Images")
    fig3, axes = plt.subplots(1, 4, figsize=(12,4))
    ms_indices = np.where(y==1)[0][:2]
    healthy_indices = np.where(y==0)[0][:2]
    indices = np.concatenate([healthy_indices, ms_indices])
    for i, ax in enumerate(axes):
        ax.imshow(X[indices[i]], cmap="gray")
        ax.set_title("Label: " + str(y[indices[i]]))
        ax.axis("off")
    st.pyplot(fig3)

    # -------------------------
    # Performance Metrics
    # -------------------------
    st.subheader("Model Performance")

    # Load SVM model and preprocessors
    svm = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")

    # Preprocess test images
    def preprocess_features(X_images):
        X_lbp = []
        X_hog = []

        for img in X_images:
            lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            X_lbp.append(hist)

            hog_features, _ = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2), visualize=True, feature_vector=True)
            X_hog.append(hog_features)

        X_features = np.hstack([X_lbp, X_hog])
        X_scaled = scaler.transform(X_features)
        X_pca = pca.transform(X_scaled)
        return X_pca

    X_test_processed = preprocess_features(X_test)

    y_pred = svm.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy on Test Set:** {accuracy:.4f}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix:")
    st.text(confusion_matrix(y_test, y_pred))

    # ROC Curve
    y_test_bin = label_binarize(y_test, classes=[0,1]).ravel()
    y_score = svm.predict_proba(X_test_processed)[:,1]
    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
    roc_auc = auc(fpr, tpr)

    fig4, ax4 = plt.subplots(figsize=(8,6))
    ax4.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    ax4.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    ax4.set_xlim([0,1])
    ax4.set_ylim([0,1.05])
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('SVM ROC Curve on Test Set')
    ax4.legend(loc="lower right")
    ax4.grid(True)
    st.pyplot(fig4)

    st.write(f"**AUC on Test Set:** {roc_auc:.4f}")

# -------------------------
# TAB 3: Prediction
# -------------------------
with tab3:
    st.markdown('<p class="section-header">MS Detection Prediction</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a brain MRI image", type=['jpg','jpeg','png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
        if st.button('Analyze Image'):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (224,224))
            X_single = np.array([img])

            # LBP + HOG features
            X_lbp = []
            X_hog = []
            for img in X_single:
                lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                X_lbp.append(hist)

                hog_features, _ = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2), visualize=True, feature_vector=True)
                X_hog.append(hog_features)

            X_features = np.hstack([X_lbp, X_hog])
            X_scaled = scaler.transform(X_features)
            X_pca = pca.transform(X_scaled)

            y_pred_single = svm.predict(X_pca)
            y_prob_single = svm.predict_proba(X_pca)[0]
            label = "(MS)" if y_pred_single[0]==1 else "(Healthy)"
            confidence = y_prob_single[y_pred_single[0]]*100

            if y_pred_single[0]==1:
                st.error(f"**Result: {label}**")
            else:
                st.success(f"**Result: {label}**")
            st.write(f"Confidence: {confidence:.2f}%")

# -------------------------
# TAB 4: Team
# -------------------------
with tab4:
    st.markdown('<p class="section-header">Team Members</p>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3, col4 = st.columns(4)

        member_box_style = """
        border:1px solid #D4BEE4; 
        border-radius:12px; 
        padding:20px; 
        margin:10px 0; 
        background:#F9F5FF;
        height:180px; 
        display:flex; 
        flex-direction:column; 
        justify-content:center;
        text-align:center;
        """

        # Member 1
        with col1:
            st.markdown(f"""
            <div style="{member_box_style}">
              <h4 style="margin:0 0 8px 0; color:#674188">Sarah Alowjan</h4>
              <p style="margin:0">Software & AI Engineer</p>
            </div>
            """, unsafe_allow_html=True)

        # Member 2
        with col2:
            st.markdown(f"""
            <div style="{member_box_style}">
              <h4 style="margin:0 0 8px 0; color:#674188">Daniyah Almusa</h4>
              <p style="margin:0"> اكتبي هنا الرول حقك</p>
            </div>
            """, unsafe_allow_html=True)

        # Member 3
        with col3:
            st.markdown(f"""
            <div style="{member_box_style}">
              <h4 style="margin:0 0 8px 0; color:#674188">Mansor Alshamran</h4>
              <p style="margin:0">AI Engineer</p>
            </div>
            """, unsafe_allow_html=True)

        # Member 4
        with col4:
            st.markdown(f"""
            <div style="{member_box_style}">
              <h4 style="margin:0 0 8px 0; color:#674188">Rayid Alshammari</h4>
              <p style="margin:0">Data Scientist</p>
            </div>
            """, unsafe_allow_html=True)

    # Contributions Section
    st.markdown('<p class="section-header">Team Contributions</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Mansor Alshamran
    - Suggested the project idea  
    - Contributed to dataset search  
    - Developed the **KNN model** as part of the comparison stage  

    ### Daniyah Almusa
    - Suggested the project idea  
    - Contributed to dataset search  
    - Implemented the **SVM model** (final chosen model after comparisons)  
    - Developed the **Prediction Page** in Streamlit (user uploads MRI images → model prediction)  
    
    ### Rayid Alshammari
    - Handled **data preprocessing**  
    - Applied **feature extraction**  
    - Conducted **feature engineering** to prepare the dataset  

    ### Sarah Alowjan
    - Collected the dataset  
    - Experimented with **Ensemble Voting Classifier**   
    - Designed the overall **Streamlit application**  
    """)

# -------------------------
# Footer
# -------------------------
import base64

logo_path = "logo.png"
with open(logo_path, "rb") as f:
    data = f.read()
encoded = base64.b64encode(data).decode()

st.markdown(f"""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #F0F8FF 0%, #E6E6FA 100%); border-radius: 16px; margin-top: 3rem; border: 2px solid #4A6FA5;'>
    <img src="data:image/png;base64,{encoded}" width="100" style="margin-bottom: 15px;">
    <h3 style='color: #4A6FA5;'>MS Detection System</h3>
    <p style='color: #6B8E23;'>Powered by AI for better healthcare</p>
    <p style='color: #4A6FA5;'>© 2025 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
