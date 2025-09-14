import plotly.graph_objects as go

# Model comparison data
model_data = [
    {"model": "Logistic Regression", "f1_score": 0.754, "is_best": False},
    {"model": "Decision Tree", "f1_score": 0.731, "is_best": False},
    {"model": "Random Forest", "f1_score": 0.762, "is_best": True},
    {"model": "Gradient Boosting", "f1_score": 0.753, "is_best": False}
]

# Extract data and abbreviate model names to fit 15 character limit
models = []
f1_scores = []
colors = []

for item in model_data:
    # Abbreviate model names to fit character limit
    model_name = item["model"]
    if model_name == "Logistic Regression":
        model_name = "Logistic Reg"
    elif model_name == "Gradient Boosting":
        model_name = "Gradient Boost"
    
    models.append(model_name)
    f1_scores.append(item["f1_score"])
    
    # Use different color for best performing model
    if item["is_best"]:
        colors.append('#DB4545')  # Bright red for best model
    else:
        colors.append('#1FB8CD')  # Strong cyan for others

# Create bar chart
fig = go.Figure()
fig.add_trace(go.Bar(
    x=models,
    y=f1_scores,
    marker_color=colors,
    showlegend=False,
    text=[f"{score:.3f}" for score in f1_scores],
    textposition='outside',
    hovertemplate='Model: %{x}<br>F1-Score: %{y:.3f}<extra></extra>'
))

fig.update_layout(
    title="ML Model Performance Comparison",
    xaxis_title="Models",
    yaxis_title="F1-Score"
)

fig.update_traces(cliponaxis=False)
fig.update_yaxes(range=[0.72, 0.78])

fig.write_image("ml_model_comparison.png")