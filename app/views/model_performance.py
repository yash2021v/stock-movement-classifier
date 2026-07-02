import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLOR_ACCENT_AMBER,
    COLOR_ACCENT_BLUE,
    COLOR_ACCENT_EMERALD,
    COLOR_ACCENT_GRAY,
    CV_RESULTS,
    LIGHT_CHART_LAYOUT,
    PLOTLY_CHART_KWARGS,
    compute_classification_report,
    compute_confusion_matrix,
    get_feature_importances,
    get_model_display_name,
    inject_base_style,
    load_model,
    load_test_predictions,
)

inject_base_style()

model_name = get_model_display_name()

st.title("Model Performance")
st.caption("5-fold TimeSeriesSplit cross-validation, pooled 20-stock data.")

st.divider()

# --- CV model comparison ---
st.subheader("Cross-Validated Model Comparison")

names = list(CV_RESULTS.keys())
means = [CV_RESULTS[n]["mean"] for n in names]
stds = [CV_RESULTS[n]["std"] for n in names]
bar_colors = [COLOR_ACCENT_GRAY, COLOR_ACCENT_BLUE, COLOR_ACCENT_AMBER, COLOR_ACCENT_EMERALD]

best_name = max((n for n in names if n != "Dummy"), key=lambda n: CV_RESULTS[n]["mean"])
tightest_name = min((n for n in names if n != "Dummy"), key=lambda n: CV_RESULTS[n]["std"])

fig_cv = go.Figure(
    go.Bar(
        x=names,
        y=means,
        error_y=dict(type="data", array=stds, visible=True, color="#6B7280"),
        marker_color=bar_colors,
        text=[f"{m:.4f}" for m in means],
        textposition="outside",
    )
)
fig_cv.update_layout(
    **LIGHT_CHART_LAYOUT,
    height=420,
    yaxis_title="CV Mean Accuracy",
    yaxis_range=[0.45, 0.56],
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig_cv, **PLOTLY_CHART_KWARGS)
st.caption(
    f"Error bars show ±1 standard deviation across folds. {best_name} has the highest "
    f"mean CV accuracy" + (f" and {tightest_name} the tightest spread" if tightest_name != best_name else "")
    + ", but none of the trained models clear a meaningful margin over the dummy baseline."
)

st.divider()

# --- Classification report ---
test_df = load_test_predictions()

if test_df.empty:
    st.warning("No saved test-prediction data found — classification report unavailable.")
else:
    report = compute_classification_report(test_df)
    period_start = test_df.index.min().date().isoformat()
    period_end = test_df.index.max().date().isoformat()

    st.subheader(f"Classification Report ({model_name}, {period_start} to {period_end} holdout)")
    st.caption(
        "Evaluated on a fixed calendar holdout (2025 onward) the model never saw during "
        "training or cross-validation — a genuine out-of-sample test."
    )

    report_rows = []
    for label in ["Underperform (0)", "Outperform (1)"]:
        m = report[label]
        report_rows.append(
            {
                "Class": label,
                "Precision": f"{m['precision']:.2f}",
                "Recall": f"{m['recall']:.2f}",
                "F1-score": f"{m['f1-score']:.2f}",
                "Support": str(int(m["support"])),
            }
        )
    report_rows.append(
        {
            "Class": "Accuracy",
            "Precision": "",
            "Recall": "",
            "F1-score": f"{report['accuracy']:.2f}",
            "Support": str(int(report["Underperform (0)"]["support"] + report["Outperform (1)"]["support"])),
        }
    )
    report_df = pd.DataFrame(report_rows).set_index("Class")
    st.dataframe(report_df, use_container_width=True)

    st.divider()

    # --- Confusion matrix ---
    st.subheader("Confusion Matrix")

    matrix = compute_confusion_matrix(test_df)

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted Underperform", "Predicted Outperform"],
            y=["Actual Underperform", "Actual Outperform"],
            colorscale=[[0, "#EFF6FF"], [1, COLOR_ACCENT_BLUE]],
            text=matrix,
            texttemplate="%{text}",
            textfont=dict(size=16, color="#1F2937"),
            showscale=False,
        )
    )
    fig_cm.update_layout(
        **LIGHT_CHART_LAYOUT,
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_autorange="reversed",
    )
    st.plotly_chart(fig_cm, **PLOTLY_CHART_KWARGS)
    st.caption("Exact counts, computed directly from actual vs. predicted labels on the holdout period.")

st.divider()

# --- Feature importance ---
st.subheader("Feature Importance")

model, feature_cols = load_model()
importances = get_feature_importances(model)

if importances is not None:
    importance_series = pd.Series(importances, index=feature_cols).sort_values()
    fig_imp = go.Figure(
        go.Bar(
            x=importance_series.values,
            y=importance_series.index,
            orientation="h",
            marker_color=COLOR_ACCENT_BLUE,
        )
    )
    fig_imp.update_layout(
        **LIGHT_CHART_LAYOUT,
        height=500,
        xaxis_title="Importance",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_imp, **PLOTLY_CHART_KWARGS)
else:
    st.info("Feature importances could not be extracted from the saved model.")
