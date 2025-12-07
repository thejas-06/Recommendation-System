# 🛒 Product Recommendation System for E-commerce

A comprehensive recommendation engine designed to help e-commerce businesses improve their shoppers' experience on the website, resulting in better **customer acquisition** and **retention**.

## 📋 Assignment Requirements

This project fulfills the following requirements:

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Popularity-based system for new customers | Part I: Uses rating counts to recommend popular products | ✅ |
| Collaborative filtering based on purchase history | Part II: SVD-based matrix factorization with correlation | ✅ |
| Content-based for new businesses without ratings | Part III: TF-IDF + K-Means clustering on product descriptions | ✅ |
| Deployed via Streamlit | `app.py` - Full Streamlit web application | ✅ |

---

## 🏗️ Project Structure

```
Recommendation-system/
├── app.py                                      # Streamlit web application
├── train_models.ipynb                          # Jupyter notebook for model training
├── product-recommendation-system-for-e-commerce.ipynb  # Original analysis notebook
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
├── ratings_Beauty.csv                          # Amazon Beauty ratings dataset
├── product_descriptions.csv                    # Product descriptions dataset
├── models/                                     # Trained models (generated)
│   ├── popular_products.pkl
│   ├── collaborative_filtering.pkl
│   └── content_based.pkl
└── venv/                                       # Virtual environment
```

---

## 🚀 Quick Start

### Step 1: Clone/Download the Project

### Step 2: Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# OR
source venv/bin/activate      # Linux/Mac
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Models
Open and run `train_models.ipynb` in Jupyter Notebook:
```bash
jupyter notebook train_models.ipynb
```
Run all cells to train and save the models.

### Step 5: Run the Streamlit App
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`

---

## 📊 Recommendation Systems Explained

### Part I: Popularity-Based System
**Target:** New customers without purchase history

**Approach:**
- Count ratings for each product
- Higher rating count = More popular
- Also tracks average rating for quality

**Use Case:** First-time visitors see the most popular products

---

### Part II: Collaborative Filtering (SVD)
**Target:** Returning customers with purchase history

**Approach:**
1. Create User-Item utility matrix
2. Apply Truncated SVD for dimensionality reduction
3. Compute product correlation matrix
4. Recommend products with high correlation (>0.8) to purchased items

**Use Case:** "Customers who bought this also bought..."

---

### Part III: Content-Based Filtering
**Target:** New businesses without any rating history

**Approach:**
1. TF-IDF vectorization of product descriptions
2. K-Means clustering to group similar products
3. Match search queries to relevant clusters

**Use Case:** Search-based recommendations for new e-commerce sites

---

## 🎨 App Features

- **Beautiful UI** with custom styling
- **Interactive controls** (sliders, dropdowns, search)
- **Visualizations** (bar charts, correlation plots)
- **Responsive design** for different screen sizes
- **Cached models** for fast loading

---

## 📁 Datasets

1. **Amazon Beauty Ratings** (`ratings_Beauty.csv`)
   - ~2 million ratings
   - Columns: UserId, ProductId, Rating, Timestamp

2. **Product Descriptions** (`product_descriptions.csv`)
   - ~124,000 products
   - Columns: product_uid, product_description

---

## 🛠️ Technologies Used

- **Python 3.11+**
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
  - TruncatedSVD
  - TfidfVectorizer
  - KMeans
- **Matplotlib** - Visualizations

---

## 📦 Deployment (Streamlit Cloud)

1. Push code to GitHub (without large CSV files)
2. Push only the `models/` folder with trained models
3. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
4. Deploy with one click!

---

## 👨‍🎓 Author

**Student Project** - Product Recommendation System for E-commerce Businesses

---

## 📝 License

This project is for educational purposes.
