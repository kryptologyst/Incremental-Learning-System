"""Streamlit demo application for incremental learning system."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models import IncrementalLearningSystem, SGDIncrementalLearner, OnlineNaiveBayes
from src.data import DataLoader, IncrementalDataGenerator
from src.metrics import IncrementalLearningMetrics, ContinualLearningEvaluator

# Page configuration
st.set_page_config(
    page_title="Incremental Learning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Incremental Learning System</h1>', unsafe_allow_html=True)
    
    # Safety disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Research & Education Use Only</h4>
    <p><strong>This system is designed for research and educational purposes only.</strong></p>
    <ul>
        <li>Not intended for production decisions or control systems</li>
        <li>Results should not be used for critical decision-making</li>
        <li>Always validate findings with domain experts</li>
        <li>Consider ethical implications of AI systems</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Dataset selection
    dataset_options = {
        "Iris": "iris",
        "Wine": "wine", 
        "Breast Cancer": "breast_cancer",
        "Synthetic": "synthetic"
    }
    selected_dataset = st.sidebar.selectbox("Dataset", list(dataset_options.keys()))
    dataset_name = dataset_options[selected_dataset]
    
    # Algorithm selection
    algorithm_options = {
        "Stochastic Gradient Descent": "sgd",
        "Online Naive Bayes": "naive_bayes"
    }
    selected_algorithm = st.sidebar.selectbox("Algorithm", list(algorithm_options.keys()))
    algorithm_name = algorithm_options[selected_algorithm]
    
    # Parameters
    st.sidebar.subheader("Parameters")
    batch_size = st.sidebar.slider("Batch Size", 5, 50, 15)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
    random_state = st.sidebar.number_input("Random Seed", 0, 1000, 42)
    
    # Experiment type
    experiment_type = st.sidebar.selectbox(
        "Experiment Type",
        ["Baseline", "Continual Learning", "Online Learning", "Concept Drift"]
    )
    
    # Main content
    if st.sidebar.button("🚀 Run Experiment"):
        run_experiment(dataset_name, algorithm_name, batch_size, test_size, random_state, experiment_type)
    
    # Information section
    st.markdown("---")
    st.markdown("### 📊 About Incremental Learning")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 What is Incremental Learning?**
        
        Incremental learning is a machine learning approach where models are trained progressively as new data arrives, allowing continuous adaptation to changing data distributions.
        """)
    
    with col2:
        st.markdown("""
        **🔄 Key Benefits**
        
        - Continuous adaptation to new patterns
        - Memory efficient (no need to retrain from scratch)
        - Suitable for streaming data scenarios
        - Reduces computational overhead
        """)
    
    with col3:
        st.markdown("""
        **⚠️ Challenges**
        
        - Catastrophic forgetting
        - Stability-plasticity dilemma
        - Concept drift handling
        - Evaluation complexity
        """)
    
    # Author information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p><strong>Author:</strong> <a href="https://github.com/kryptologyst" target="_blank">kryptologyst</a></p>
    <p><strong>GitHub:</strong> <a href="https://github.com/kryptologyst" target="_blank">https://github.com/kryptologyst</a></p>
    </div>
    """, unsafe_allow_html=True)


def run_experiment(dataset_name: str, algorithm_name: str, batch_size: int, 
                  test_size: float, random_state: int, experiment_type: str):
    """Run the selected experiment and display results."""
    
    with st.spinner("Running experiment..."):
        
        # Load data
        data_loader = DataLoader(
            dataset_name=dataset_name,
            test_size=test_size,
            random_state=random_state
        )
        
        X_train, X_test, y_train, y_test = data_loader.load_dataset()
        
        # Create model
        model = IncrementalLearningSystem(
            algorithm=algorithm_name,
            random_state=random_state
        )
        
        if experiment_type == "Baseline":
            run_baseline_experiment(model, X_train, y_train, X_test, y_test, batch_size)
        elif experiment_type == "Continual Learning":
            run_continual_learning_experiment(model, random_state)
        elif experiment_type == "Online Learning":
            run_online_learning_experiment(model, random_state, batch_size)
        elif experiment_type == "Concept Drift":
            run_concept_drift_experiment(model, X_train, y_train, X_test, y_test, batch_size)


def run_baseline_experiment(model, X_train, y_train, X_test, y_test, batch_size):
    """Run baseline incremental learning experiment."""
    
    st.subheader("📈 Baseline Incremental Learning Results")
    
    # Train incrementally
    accuracies = model.train_incremental(
        X_train, y_train,
        batch_size=batch_size,
        classes=np.unique(y_train)
    )
    
    # Evaluate on test set
    test_metrics = model.evaluate(X_test, y_test)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Accuracy", f"{test_metrics['accuracy']:.4f}")
    
    with col2:
        f1_score = test_metrics['classification_report']['macro avg']['f1-score']
        st.metric("F1 Score", f"{f1_score:.4f}")
    
    with col3:
        precision = test_metrics['classification_report']['macro avg']['precision']
        st.metric("Precision", f"{precision:.4f}")
    
    with col4:
        recall = test_metrics['classification_report']['macro avg']['recall']
        st.metric("Recall", f"{recall:.4f}")
    
    # Plot learning curve
    if accuracies:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(accuracies) + 1)),
            y=accuracies,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig.update_layout(
            title="Learning Curve - Accuracy Over Batches",
            xaxis_title="Batch Number",
            yaxis_title="Accuracy",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix
    y_pred = model.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        color_continuous_scale="Blues"
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)


def run_continual_learning_experiment(model, random_state):
    """Run continual learning experiment."""
    
    st.subheader("🔄 Continual Learning Results")
    
    # Generate sequential tasks
    data_generator = IncrementalDataGenerator(random_state=random_state)
    tasks = data_generator.generate_sequential_tasks(n_tasks=3, samples_per_task=200)
    
    # Create test tasks
    test_tasks = []
    for X_train, y_train in tasks:
        test_size = int(0.2 * len(X_train))
        X_test = X_train[-test_size:]
        y_test = y_train[-test_size:]
        test_tasks.append((X_test, y_test))
    
    # Evaluate
    evaluator = ContinualLearningEvaluator()
    results = evaluator.evaluate_task_sequence(model, tasks, test_tasks)
    
    # Display forgetting metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Backward Transfer", f"{results['forgetting_metrics']['backward_transfer']:.4f}")
    
    with col2:
        st.metric("Forward Transfer", f"{results['forgetting_metrics']['forward_transfer']:.4f}")
    
    with col3:
        st.metric("Average Accuracy", f"{results['forgetting_metrics']['average_accuracy']:.4f}")
    
    # Plot task accuracy matrix
    task_accuracies = np.array(results['task_accuracies'])
    
    fig = px.imshow(
        task_accuracies,
        text_auto=True,
        aspect="auto",
        title="Task Accuracy Matrix",
        labels=dict(x="Test Task", y="Training Task", color="Accuracy"),
        color_continuous_scale="RdYlBu_r"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def run_online_learning_experiment(model, random_state, batch_size):
    """Run online learning experiment."""
    
    st.subheader("🌊 Online Learning Results")
    
    # Generate non-stationary data
    data_generator = IncrementalDataGenerator(random_state=random_state)
    X_stream, y_stream = data_generator.generate_non_stationary_data(n_samples=1000, drift_frequency=200)
    
    # Evaluate
    evaluator = ContinualLearningEvaluator()
    results = evaluator.evaluate_online_learning(
        model, X_stream, y_stream,
        batch_size=batch_size,
        evaluation_frequency=5
    )
    
    # Display efficiency metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Learning Speed", f"{results['efficiency_metrics']['learning_speed']:.6f}")
    
    with col2:
        st.metric("Sample Efficiency", f"{results['efficiency_metrics']['sample_efficiency']:.0f}")
    
    with col3:
        st.metric("AULC", f"{results['efficiency_metrics']['area_under_learning_curve']:.2f}")
    
    # Plot learning curve
    if results['accuracies']:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['sample_counts'],
            y=results['accuracies'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#ff7f0e', width=3)
        ))
        
        fig.update_layout(
            title="Online Learning Curve",
            xaxis_title="Samples Seen",
            yaxis_title="Accuracy",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def run_concept_drift_experiment(model, X_train, y_train, X_test, y_test, batch_size):
    """Run concept drift experiment."""
    
    st.subheader("🌊 Concept Drift Results")
    
    # Simulate concept drift
    data_loader = DataLoader()
    X_drifted, y_drifted = data_loader.simulate_concept_drift(X_train, y_train, drift_point=0.5)
    
    # Train on original data
    model.train_incremental(X_train, y_train, batch_size=batch_size, classes=np.unique(y_train))
    original_accuracy = model.score(X_test, y_test)
    
    # Adapt to drifted data
    model.train_incremental(X_drifted, y_drifted, batch_size=batch_size)
    adapted_accuracy = model.score(X_test, y_test)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Original Accuracy", f"{original_accuracy:.4f}")
    
    with col2:
        st.metric("Adapted Accuracy", f"{adapted_accuracy:.4f}")
    
    # Plot accuracy comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Original", "After Drift"],
        y=[original_accuracy, adapted_accuracy],
        marker_color=['#1f77b4', '#ff7f0e']
    ))
    
    fig.update_layout(
        title="Accuracy Before and After Concept Drift",
        yaxis_title="Accuracy",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
