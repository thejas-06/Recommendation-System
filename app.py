"""
=============================================================
Product Recommendation System - Streamlit Web Application
=============================================================

Student Name: Thejas AN
Course: Data Science
Assignment: Product Recommendation System for E-commerce

This app provides 3 types of product recommendations:
1. Popularity-Based (for new customers)
2. Collaborative Filtering (for returning customers)  
3. Content-Based (for new businesses)

How to Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1E88E5;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        padding: 0.5rem 0;
        border-left: 4px solid #1E88E5;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .recommendation-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load Models
# ============================================================
@st.cache_data
def load_popularity_model():
    """Load popularity-based model"""
    try:
        return pd.read_pickle('models/popular_products.pkl')
    except FileNotFoundError:
        return None

@st.cache_data
def load_collaborative_model():
    """Load collaborative filtering model (optimized - no full correlation matrix)"""
    try:
        with open('models/collaborative_filtering.pkl', 'rb') as f:
            data = pickle.load(f)
        # Check if using optimized model (no correlation_matrix) or legacy model
        if 'correlation_matrix' not in data:
            # Optimized model - we'll compute correlations at runtime
            return data
        else:
            # Legacy model with full correlation matrix
            return data
    except FileNotFoundError:
        return None

def compute_correlation_for_product(decomposed_matrix, product_idx):
    """Compute correlation between one product and all others at runtime.
    This saves ~60MB of storage by not pre-computing the full correlation matrix.
    """
    # Get the product's decomposed vector
    product_vector = decomposed_matrix[product_idx]
    
    # Compute correlation with all products using cosine similarity
    # (which is equivalent to Pearson correlation for centered/normalized data)
    norms = np.linalg.norm(decomposed_matrix, axis=1)
    product_norm = norms[product_idx]
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        correlations = np.dot(decomposed_matrix, product_vector) / (norms * product_norm)
        correlations = np.nan_to_num(correlations, nan=0.0)
    
    return correlations

@st.cache_data
def load_content_model():
    """Load content-based model"""
    try:
        with open('models/content_based.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart--v1.png", width=80)
    st.title("Navigation")
    
    page = st.radio(
        "Select Recommendation Type:",
        ["🏠 Home", 
         "📈 Part I: Popularity-Based",
         "🤝 Part II: Collaborative Filtering",
         "📝 Part III: Content-Based"],
        index=0
    )
    
    st.divider()
    
    st.markdown("### 📋 About This Project")
    st.info("""
    **Product Recommendation System**
    
    A machine learning project that helps 
    e-commerce businesses improve customer 
    experience through smart recommendations.
    
    **Three Approaches:**
    1. 📈 Popularity-based
    2. 🤝 Collaborative filtering  
    3. 📝 Content-based
    """)
    
    st.divider()
    
    # Student Information - Add your details!
    st.markdown("### 👨‍🎓 Student Info")
    st.markdown("""
    **Name:** Thejas AN  
    **Course:** Data Science
    """)
    
    st.divider()
    st.markdown("##### Built with ❤️ using Streamlit")

# ============================================================
# Home Page
# ============================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">🛒 Product Recommendation System</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome!
    
    This recommendation system helps e-commerce businesses improve their shoppers' experience,
    resulting in better **customer acquisition** and **retention**.
    
    ---
    
    ### 📊 Three Recommendation Approaches
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; height: 280px;">
            <h3>📈 Part I</h3>
            <h4>Popularity-Based</h4>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p><b>Target:</b> New Customers</p>
            <p>Recommends most popular products based on rating counts for first-time visitors.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; height: 280px;">
            <h3>🤝 Part II</h3>
            <h4>Collaborative Filtering</h4>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p><b>Target:</b> Returning Customers</p>
            <p>Uses SVD to recommend products based on similar users' preferences.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; height: 280px;">
            <h3>📝 Part III</h3>
            <h4>Content-Based</h4>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p><b>Target:</b> New Businesses</p>
            <p>Uses text clustering on product descriptions for businesses without ratings.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🚀 How It Works")
    st.markdown("""
    1. **New Customer Lands** → Show popular products (Part I)
    2. **Customer Browses/Purchases** → Use collaborative filtering (Part II)
    3. **No Rating History Available** → Use content-based filtering (Part III)
    
    ---
    
    ### 👉 Get Started
    Select a recommendation type from the **sidebar** to explore!
    """)

# ============================================================
# Part I: Popularity-Based Recommendations
# ============================================================
elif page == "📈 Part I: Popularity-Based":
    st.markdown('<p class="main-header">📈 Popularity-Based Recommendations</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Business Context
    When a **new customer** visits the e-commerce website for the first time without any purchase history,
    we recommend the **most popular products** sold on the website.
    
    ---
    """)
    
    # Load model
    popular_products = load_popularity_model()
    
    if popular_products is None:
        st.error("⚠️ Model not found! Please run `train_models.ipynb` first to train the models.")
        st.code("jupyter notebook train_models.ipynb", language="bash")
    else:
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Products", f"{len(popular_products):,}")
        with col2:
            st.metric("Top Product Ratings", f"{popular_products['RatingCount'].max():,}")
        with col3:
            st.metric("Avg Rating (Top 10)", f"{popular_products.head(10)['AvgRating'].mean():.2f}")
        
        st.markdown("---")
        
        # Controls
        col1, col2 = st.columns([1, 3])
        with col1:
            n_recommendations = st.slider(
                "Number of Recommendations",
                min_value=5,
                max_value=50,
                value=10,
                step=5
            )
        
        # Display recommendations
        st.markdown(f'<p class="section-header">🏆 Top {n_recommendations} Popular Products</p>', 
                   unsafe_allow_html=True)
        
        top_products = popular_products.head(n_recommendations).reset_index()
        top_products.columns = ['Product ID', 'Rating Count', 'Avg Rating']
        top_products['Rank'] = range(1, n_recommendations + 1)
        top_products = top_products[['Rank', 'Product ID', 'Rating Count', 'Avg Rating']]
        top_products['Avg Rating'] = top_products['Avg Rating'].round(2)
        
        st.dataframe(
            top_products,
            use_container_width=True,
            hide_index=True
        )
        
        # Visualization
        st.markdown("### 📊 Popularity Visualization")
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_recommendations))[::-1]
        bars = ax.bar(range(n_recommendations), top_products['Rating Count'], color=colors)
        ax.set_xlabel('Product Rank', fontsize=12)
        ax.set_ylabel('Number of Ratings', fontsize=12)
        ax.set_title(f'Top {n_recommendations} Most Popular Products', fontsize=14, fontweight='bold')
        ax.set_xticks(range(n_recommendations))
        ax.set_xticklabels([f"#{i+1}" for i in range(n_recommendations)])
        
        # Add value labels on bars
        for bar, count in zip(bars, top_products['Rating Count']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                   f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================
# Part II: Collaborative Filtering
# ============================================================
elif page == "🤝 Part II: Collaborative Filtering":
    st.markdown('<p class="main-header">🤝 Collaborative Filtering Recommendations</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Business Context
    Once a customer makes a purchase, we recommend products based on:
    - Their **purchase history**
    - **Ratings provided by other users** who bought similar items
    
    ### 🔬 Methodology: SVD (Singular Value Decomposition)
    - Creates a User-Item utility matrix
    - Applies dimensionality reduction
    - Computes product correlations for recommendations
    
    ---
    """)
    
    # Load model
    collab_data = load_collaborative_model()
    
    if collab_data is None:
        st.error("⚠️ Model not found! Please run `train_models.ipynb` first to train the models.")
        st.code("jupyter notebook train_models.ipynb", language="bash")
    else:
        product_names = collab_data['product_names']
        decomposed_matrix = collab_data['decomposed_matrix']
        # Check if using legacy model with pre-computed correlations
        use_legacy = 'correlation_matrix' in collab_data
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Products in Model", f"{len(product_names):,}")
        with col2:
            st.metric("SVD Components", f"{collab_data['decomposed_matrix'].shape[1]}")
        
        st.markdown("---")
        
        # Product selection
        st.markdown('<p class="section-header">🔍 Select a Product</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_product = st.selectbox(
                "Choose a product the customer has purchased:",
                product_names,
                index=50
            )
        
        with col2:
            correlation_threshold = st.slider(
                "Minimum Correlation",
                min_value=0.5,
                max_value=0.99,
                value=0.8,
                step=0.05
            )
        
        # Get recommendations
        if selected_product:
            product_idx = product_names.index(selected_product)
            # Compute correlation at runtime (optimized) or use pre-computed (legacy)
            if use_legacy:
                correlation_values = collab_data['correlation_matrix'][product_idx]
            else:
                correlation_values = compute_correlation_for_product(decomposed_matrix, product_idx)
            
            # Find similar products
            similar_indices = np.where(correlation_values > correlation_threshold)[0]
            similar_products = [
                (product_names[i], correlation_values[i])
                for i in similar_indices
                if i != product_idx
            ]
            similar_products.sort(key=lambda x: x[1], reverse=True)
            
            st.markdown(f'<p class="section-header">🎁 Recommended Products (Correlation > {correlation_threshold})</p>', 
                       unsafe_allow_html=True)
            
            if similar_products:
                # Create dataframe
                rec_df = pd.DataFrame(similar_products, columns=['Product ID', 'Correlation Score'])
                rec_df['Rank'] = range(1, len(rec_df) + 1)
                rec_df['Correlation Score'] = rec_df['Correlation Score'].round(4)
                rec_df = rec_df[['Rank', 'Product ID', 'Correlation Score']]
                
                st.success(f"Found **{len(similar_products)}** similar products!")
                st.dataframe(rec_df.head(15), use_container_width=True, hide_index=True)
                
                # Visualization
                if len(similar_products) > 0:
                    st.markdown("### 📊 Correlation Scores")
                    
                    import matplotlib.pyplot as plt
                    
                    top_n = min(10, len(similar_products))
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    products = [p[0][:15] + "..." if len(p[0]) > 15 else p[0] for p in similar_products[:top_n]]
                    scores = [p[1] for p in similar_products[:top_n]]
                    
                    colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, top_n))
                    bars = ax.barh(range(top_n), scores, color=colors)
                    ax.set_yticks(range(top_n))
                    ax.set_yticklabels(products)
                    ax.set_xlabel('Correlation Score', fontsize=12)
                    ax.set_title('Top Similar Products by Correlation', fontsize=14, fontweight='bold')
                    ax.invert_yaxis()
                    ax.set_xlim(correlation_threshold, 1.0)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.warning(f"No products found with correlation > {correlation_threshold}. Try lowering the threshold.")

# ============================================================
# Part III: Content-Based Filtering
# ============================================================
elif page == "📝 Part III: Content-Based":
    st.markdown('<p class="main-header">📝 Content-Based Recommendations</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Business Context
    For a business **setting up its e-commerce website for the first time** without any product ratings,
    we use **product descriptions** to recommend similar items.
    
    ### 🔬 Methodology: TF-IDF + K-Means Clustering
    - Convert product descriptions to TF-IDF vectors
    - Cluster similar products using K-Means
    - Match user search queries to relevant clusters
    
    ---
    """)
    
    # Load model
    content_data = load_content_model()
    
    if content_data is None:
        st.error("⚠️ Model not found! Please run `train_models.ipynb` first to train the models.")
        st.code("jupyter notebook train_models.ipynb", language="bash")
    else:
        vectorizer = content_data['vectorizer']
        kmeans = content_data['kmeans_model']
        cluster_descriptions = content_data['cluster_descriptions']
        n_clusters = content_data['n_clusters']
        product_data = content_data['product_data']
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Products", f"{len(product_data):,}")
        with col2:
            st.metric("Clusters", n_clusters)
        with col3:
            st.metric("Vocabulary Size", f"{len(vectorizer.get_feature_names_out()):,}")
        
        st.markdown("---")
        
        # Search interface
        st.markdown('<p class="section-header">🔍 Search for Products</p>', unsafe_allow_html=True)
        
        search_query = st.text_input(
            "Enter search keywords:",
            placeholder="e.g., cutting tool, water heater, light bulb, outdoor furniture"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔍 Get Recommendations", type="primary"):
                if search_query:
                    # Transform query and predict cluster
                    Y = vectorizer.transform([search_query])
                    predicted_cluster = kmeans.predict(Y)[0]
                    
                    st.session_state['predicted_cluster'] = predicted_cluster
                    st.session_state['search_done'] = True
        
        # Display results
        if st.session_state.get('search_done') and search_query:
            predicted_cluster = st.session_state['predicted_cluster']
            
            st.markdown(f'<p class="section-header">🎯 Results for "{search_query}"</p>', 
                       unsafe_allow_html=True)
            
            st.info(f"**Matched Cluster:** {predicted_cluster}")
            
            # Show related terms
            st.markdown("### 📌 Related Product Terms")
            terms = cluster_descriptions[predicted_cluster]
            
            # Display as tags
            term_html = " ".join([
                f'<span style="background-color: #E3F2FD; padding: 5px 10px; margin: 3px; '
                f'border-radius: 15px; display: inline-block; color: #1565C0;">{term}</span>'
                for term in terms
            ])
            st.markdown(term_html, unsafe_allow_html=True)
            
            # Show products in cluster
            st.markdown("### 🛒 Products in This Category")
            
            cluster_products = product_data[kmeans.labels_ == predicted_cluster]
            
            for idx, row in cluster_products.head(5).iterrows():
                with st.expander(f"Product ID: {row['product_uid']}"):
                    st.write(row['product_description'][:500] + "...")
        
        # Cluster overview
        st.markdown("---")
        st.markdown('<p class="section-header">📊 All Product Clusters</p>', unsafe_allow_html=True)
        
        # Create cluster summary
        cluster_summary = []
        for i in range(n_clusters):
            count = (kmeans.labels_ == i).sum()
            top_terms = ", ".join(cluster_descriptions[i][:5])
            cluster_summary.append({
                'Cluster': i,
                'Products': count,
                'Top Keywords': top_terms
            })
        
        cluster_df = pd.DataFrame(cluster_summary)
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)
        
        # Visualization
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 5))
        counts = [cluster_summary[i]['Products'] for i in range(n_clusters)]
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        ax.bar(range(n_clusters), counts, color=colors)
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Number of Products', fontsize=12)
        ax.set_title('Product Distribution Across Clusters', fontsize=14, fontweight='bold')
        ax.set_xticks(range(n_clusters))
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>📧 Product Recommendation System | Built for E-commerce Businesses</p>
    <p style="font-size: 0.8rem;">Powered by Machine Learning: SVD, TF-IDF, K-Means Clustering</p>
</div>
""", unsafe_allow_html=True)
