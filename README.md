<div align="center">

# 🛒 Product Recommendation System for E-Commerce

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**A comprehensive machine learning-powered recommendation engine designed to enhance e-commerce user experience through intelligent product suggestions.**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [License](#-license)

</div>

---

## 📋 Overview

This project implements a **multi-strategy product recommendation system** that helps e-commerce businesses improve customer acquisition and retention by providing personalized product suggestions. The system employs three distinct recommendation approaches, each tailored for different business scenarios.

### Key Highlights

- 🎯 **Three recommendation strategies** for different user contexts
- 📊 **Interactive web interface** built with Streamlit
- 🤖 **Machine learning models** using SVD, TF-IDF, and K-Means
- 📈 **Scalable architecture** with pre-trained model caching
- 📱 **Responsive design** for various screen sizes

---

## ✨ Features

| Approach | Target Audience | Algorithm | Use Case |
|----------|----------------|-----------|----------|
| **Popularity-Based** | New Customers | Rating Count Analysis | Recommend trending products to first-time visitors |
| **Collaborative Filtering** | Returning Customers | Truncated SVD | "Customers who bought this also bought..." |
| **Content-Based** | New Businesses | TF-IDF + K-Means | Search-based recommendations without rating history |

---

## 🎬 Demo

### Home Page
The home page provides an overview of all three recommendation approaches with interactive navigation.

### Popularity-Based Recommendations
Displays the most popular products ranked by rating count with interactive visualizations.

### Collaborative Filtering
Select a product to find similar items based on user behavior patterns and purchase correlations.

### Content-Based Search
Search for products using text queries to find similar items based on product descriptions.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Product Recommendation System                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐ │
│  │   Part I        │  │    Part II       │  │        Part III             │ │
│  │   Popularity    │  │  Collaborative   │  │      Content-Based          │ │
│  │     Based       │  │    Filtering     │  │                             │ │
│  ├─────────────────┤  ├──────────────────┤  ├─────────────────────────────┤ │
│  │ Rating Count    │  │ Truncated SVD    │  │ TF-IDF Vectorization        │ │
│  │ Analysis        │  │ Matrix Factor.   │  │ K-Means Clustering          │ │
│  └────────┬────────┘  └────────┬─────────┘  └──────────────┬──────────────┘ │
│           │                    │                           │                │
│           ▼                    ▼                           ▼                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                     Streamlit Web Application                           ││
│  │                        (app.py - 585 lines)                             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Recommendation-system/
│
├── 📄 app.py                                        # Streamlit web application
├── 📓 train_models.ipynb                            # Model training notebook
├── 📓 product-recommendation-system-for-e-commerce.ipynb  # Analysis notebook
├── 📄 requirements.txt                              # Python dependencies
├── 📄 README.md                                     # Project documentation
├── 📄 LICENSE                                       # MIT License
├── 📄 .gitignore                                    # Git ignore rules
│
├── 📂 models/                                       # Pre-trained models
│   ├── popular_products.pkl                         # Popularity rankings
│   ├── collaborative_filtering.pkl                  # SVD decomposition
│   └── content_based.pkl                            # TF-IDF + K-Means
│
└── 📂 data/ (not included - see Data section)
    ├── ratings_Beauty.csv                           # Amazon Beauty ratings
    └── product_descriptions.csv                     # Product descriptions
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Recommendation-system.git
cd Recommendation-system
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train Models (First Time Only)

> ⚠️ **Note:** You need the datasets (`ratings_Beauty.csv` and `product_descriptions.csv`) to train the models.

```bash
jupyter notebook train_models.ipynb
```

Run all cells to train and save the models to the `models/` directory.

### Step 5: Launch the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📊 Usage

### 1. Popularity-Based Recommendations (Part I)

**Best for:** New customers without purchase history

- Navigate to "Part I: Popularity-Based" in the sidebar
- Adjust the slider to view top N products
- Products are ranked by rating count (more ratings = more popular)

### 2. Collaborative Filtering (Part II)

**Best for:** Returning customers with purchase history

- Navigate to "Part II: Collaborative Filtering"
- Select a product the customer has previously purchased
- Adjust the correlation threshold (default: 0.8)
- View products with similar user purchase patterns

### 3. Content-Based Filtering (Part III)

**Best for:** New businesses without rating data

- Navigate to "Part III: Content-Based"
- Enter search keywords (e.g., "cutting tool", "water heater")
- Click "Get Recommendations"
- View products from matching clusters

---

## 📈 Datasets

This project uses Amazon product data:

| Dataset | Description | Size |
|---------|-------------|------|
| `ratings_Beauty.csv` | User ratings for beauty products | ~2M ratings |
| `product_descriptions.csv` | Product descriptions | ~124K products |

**Data Schema:**

```
ratings_Beauty.csv
├── UserId      : Unique user identifier
├── ProductId   : Unique product ASIN
├── Rating      : Rating value (1-5)
└── Timestamp   : Unix timestamp

product_descriptions.csv
├── product_uid        : Unique product identifier
└── product_description: Text description of product
```

---

## 🔬 Technical Details

### Recommendation Algorithms

#### Part I: Popularity-Based
```
Popular Products = Count(Ratings) per Product
                   ↓
            Sort by Count (Descending)
                   ↓
            Return Top N Products
```

#### Part II: Collaborative Filtering (Matrix Factorization)
```
User-Item Matrix → Truncated SVD → Decomposed Matrix
                                         ↓
                              Cosine Similarity
                                         ↓
                           Correlation Threshold > 0.8
                                         ↓
                              Similar Products
```

#### Part III: Content-Based (Text Clustering)
```
Product Descriptions → TF-IDF Vectorization → K-Means Clustering
                                                      ↓
         User Query → TF-IDF Transform → Predict Cluster
                                                      ↓
                                    Return Products in Cluster
```

---

## 🛠️ Technologies

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11+ |
| **Web Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn (TruncatedSVD, TfidfVectorizer, KMeans) |
| **Visualization** | Matplotlib |
| **Model Persistence** | Pickle |

---

## 🚢 Deployment

### Streamlit Cloud

1. Push your code to GitHub (models included, datasets excluded via `.gitignore`)
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy with one click!

### Docker (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Thejas AN**  
Data Science Student

---

## 🙏 Acknowledgments

- Amazon Product Dataset for providing the beauty ratings data
- Streamlit team for the amazing web framework
- scikit-learn contributors for machine learning tools

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Built with ❤️ using Python and Streamlit

</div>
