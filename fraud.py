"""
Fraud Detection System using Machine Learning & AI
Author: Malla Babi Durga Jogiraju
Project: Credit Card Fraud Detection
Tech Stack: Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. DATA GENERATION (Simulated Transaction Data)
# ─────────────────────────────────────────────

def generate_transaction_data(n_samples=10000, fraud_ratio=0.02, random_state=42):
    """
    Simulates realistic credit card transaction data.
    Features mimic real-world patterns used in fraud detection.
    """
    np.random.seed(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit = pd.DataFrame({
        'transaction_amount': np.random.exponential(scale=80, size=n_legit),
        'transaction_hour':   np.random.choice(range(8, 22), size=n_legit),
        'merchant_category':  np.random.choice(['grocery', 'fuel', 'restaurant', 'retail', 'online'], size=n_legit),
        'distance_from_home': np.random.normal(loc=10, scale=5, size=n_legit).clip(0),
        'num_transactions_today': np.random.poisson(lam=3, size=n_legit),
        'avg_monthly_spend':  np.random.normal(loc=2000, scale=500, size=n_legit).clip(500),
        'is_foreign_transaction': np.random.choice([0, 1], p=[0.95, 0.05], size=n_legit),
        'is_new_merchant':    np.random.choice([0, 1], p=[0.85, 0.15], size=n_legit),
        'card_present':       np.random.choice([0, 1], p=[0.2, 0.8], size=n_legit),
        'label': 0
    })

    # Fraudulent transactions (different distribution)
    fraud = pd.DataFrame({
        'transaction_amount': np.random.exponential(scale=400, size=n_fraud),
        'transaction_hour':   np.random.choice(range(0, 6), size=n_fraud),   # odd hours
        'merchant_category':  np.random.choice(['online', 'retail'], size=n_fraud),
        'distance_from_home': np.random.normal(loc=100, scale=50, size=n_fraud).clip(0),
        'num_transactions_today': np.random.poisson(lam=8, size=n_fraud),    # many txns
        'avg_monthly_spend':  np.random.normal(loc=2000, scale=500, size=n_fraud).clip(500),
        'is_foreign_transaction': np.random.choice([0, 1], p=[0.4, 0.6], size=n_fraud),
        'is_new_merchant':    np.random.choice([0, 1], p=[0.3, 0.7], size=n_fraud),
        'card_present':       np.random.choice([0, 1], p=[0.7, 0.3], size=n_fraud),
        'label': 1
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=random_state)
    df['merchant_category'] = pd.Categorical(df['merchant_category']).codes
    return df


# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_data(df):
    """Cleans data, handles class imbalance using oversampling."""
    print("=" * 55)
    print("  FRAUD DETECTION SYSTEM — Data Preprocessing")
    print("=" * 55)
    print(f"\nTotal records    : {len(df):,}")
    print(f"Legitimate txns  : {(df['label'] == 0).sum():,}")
    print(f"Fraudulent txns  : {(df['label'] == 1).sum():,}")
    print(f"Fraud rate       : {df['label'].mean() * 100:.2f}%")

    # Check for nulls
    print(f"\nMissing values   : {df.isnull().sum().sum()}")

    # Features & target
    X = df.drop('label', axis=1)
    y = df['label']

    # Handle class imbalance — oversample minority class
    df_majority = df[df['label'] == 0]
    df_minority = df[df['label'] == 1]
    df_minority_upsampled = resample(
        df_minority, replace=True,
        n_samples=len(df_majority),
        random_state=42
    )
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    X_bal = df_balanced.drop('label', axis=1)
    y_bal = df_balanced['label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"\nTraining samples : {X_train_scaled.shape[0]:,}")
    print(f"Testing samples  : {X_test_scaled.shape[0]:,}")
    print("\nPreprocessing complete.\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────

def train_models(X_train, y_train):
    """Trains 3 ML models and returns all."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    print("Training models...")
    trained = {}
    for name, model in models.items():
        print(f"  → {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
    print("All models trained.\n")
    return trained


# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────

def evaluate_models(models, X_test, y_test):
    """Evaluates all models and prints metrics."""
    print("=" * 55)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 55)
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)

        results[name] = {
            'accuracy':  acc,  'precision': prec,
            'recall':    rec,  'f1_score':  f1,
            'roc_auc':   auc,  'y_pred':    y_pred,
            'y_prob':    y_prob
        }

        print(f"\n{'─'*40}")
        print(f"  {name}")
        print(f"{'─'*40}")
        print(f"  Accuracy  : {acc  * 100:.2f}%")
        print(f"  Precision : {prec * 100:.2f}%")
        print(f"  Recall    : {rec  * 100:.2f}%")
        print(f"  F1 Score  : {f1   * 100:.2f}%")
        print(f"  ROC-AUC   : {auc  * 100:.2f}%")

    return results


# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_results(models, results, X_test, y_test, feature_names, df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Fraud Detection System — Model Analysis', fontsize=16, fontweight='bold', y=1.01)
    colors = ['#2563EB', '#16A34A', '#DC2626']

    # 1. Class distribution
    ax = axes[0, 0]
    counts = df['label'].value_counts()
    ax.bar(['Legitimate', 'Fraudulent'], counts.values, color=['#22C55E', '#EF4444'], edgecolor='white', linewidth=1.5)
    ax.set_title('Transaction Class Distribution', fontweight='bold')
    ax.set_ylabel('Count')
    for bar, val in zip(ax.patches, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', fontsize=10, fontweight='bold')

    # 2. Model accuracy comparison
    ax = axes[0, 1]
    names = list(results.keys())
    accs  = [results[n]['accuracy'] * 100 for n in names]
    short = ['Log. Reg.', 'Rand. Forest', 'Grad. Boost']
    bars  = ax.bar(short, accs, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title('Model Accuracy Comparison', fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim([80, 100])
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # 3. ROC Curves
    ax = axes[0, 2]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={res['roc_auc']:.3f})")
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_title('ROC Curves', fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=8)

    # 4. Confusion Matrix (best model = Random Forest)
    ax = axes[1, 0]
    best_name = max(results, key=lambda n: results[n]['roc_auc'])
    cm = confusion_matrix(y_test, results[best_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'])
    ax.set_title(f'Confusion Matrix\n({best_name})', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    # 5. Feature Importance (Random Forest)
    ax = axes[1, 1]
    rf = models['Random Forest']
    imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)
    imp.plot(kind='barh', ax=ax, color='#2563EB')
    ax.set_title('Feature Importance\n(Random Forest)', fontweight='bold')
    ax.set_xlabel('Importance Score')

    # 6. Metrics comparison
    ax = axes[1, 2]
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    x = np.arange(len(metrics))
    width = 0.25
    for i, (name, res) in enumerate(results.items()):
        vals = [res[m] for m in metrics]
        ax.bar(x + i*width, vals, width, label=name.split()[0], color=colors[i], alpha=0.85)
    ax.set_title('All Metrics Comparison', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Accuracy','Precision','Recall','F1','AUC'], fontsize=8)
    ax.set_ylim([0.7, 1.0])
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('fraud_detection_results.png', dpi=150, bbox_inches='tight')
    print("\nChart saved → fraud_detection_results.png")
    plt.show()


# ─────────────────────────────────────────────
# 6. PREDICTION FUNCTION (Inference / AI Layer)
# ─────────────────────────────────────────────

def predict_transaction(model, scaler, transaction: dict) -> dict:
    """
    Takes a single transaction dict and returns fraud prediction.
    This is the AI inference layer — usable in an API or app.
    """
    feature_order = [
        'transaction_amount', 'transaction_hour', 'merchant_category',
        'distance_from_home', 'num_transactions_today', 'avg_monthly_spend',
        'is_foreign_transaction', 'is_new_merchant', 'card_present'
    ]
    X = pd.DataFrame([transaction])[feature_order]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    risk = "HIGH RISK" if prob > 0.7 else "MEDIUM RISK" if prob > 0.4 else "LOW RISK"
    return {
        'prediction':    'FRAUD'    if pred == 1 else 'LEGITIMATE',
        'fraud_probability': round(float(prob) * 100, 2),
        'risk_level':    risk,
        'action':        'BLOCK TRANSACTION' if pred == 1 else 'APPROVE TRANSACTION'
    }


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Generate data
    df = generate_transaction_data(n_samples=10000, fraud_ratio=0.02)

    # Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

    # Train
    models = train_models(X_train, y_train)

    # Evaluate
    results = evaluate_models(models, X_test, y_test)

    # Plot
    plot_results(models, results, X_test, y_test, feature_names, df)

    # Demo prediction
    print("\n" + "=" * 55)
    print("  LIVE TRANSACTION PREDICTION DEMO")
    print("=" * 55)

    sample_transactions = [
        {
            "desc": "Normal grocery purchase",
            "transaction_amount": 65, "transaction_hour": 14,
            "merchant_category": 0, "distance_from_home": 5,
            "num_transactions_today": 2, "avg_monthly_spend": 1800,
            "is_foreign_transaction": 0, "is_new_merchant": 0, "card_present": 1
        },
        {
            "desc": "Suspicious large online purchase at 3 AM",
            "transaction_amount": 950, "transaction_hour": 3,
            "merchant_category": 4, "distance_from_home": 250,
            "num_transactions_today": 9, "avg_monthly_spend": 1800,
            "is_foreign_transaction": 1, "is_new_merchant": 1, "card_present": 0
        }
    ]

    best_model = models['Random Forest']
    for txn in sample_transactions:
        desc = txn.pop("desc")
        result = predict_transaction(best_model, scaler, txn)
        print(f"\nTransaction : {desc}")
        print(f"Prediction  : {result['prediction']}")
        print(f"Fraud Prob  : {result['fraud_probability']}%")
        print(f"Risk Level  : {result['risk_level']}")
        print(f"Action      : {result['action']}")