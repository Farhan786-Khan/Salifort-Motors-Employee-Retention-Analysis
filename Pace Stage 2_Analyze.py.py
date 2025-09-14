import plotly.graph_objects as go
import plotly.express as px

# Data for cost analysis
cost_data = {
    "current_annual_cost": 430000000,
    "potential_savings": 107000000,
    "remaining_cost": 323000000
}

# Create donut chart data
labels = ['Potential Savings', 'Remaining Cost']
values = [cost_data['potential_savings'], cost_data['remaining_cost']]
colors = ['#1FB8CD', '#2E8B57']  # Using brand colors

# Create donut chart
fig = go.Figure(data=[go.Pie(
    labels=labels,
    values=values,
    hole=0.5,
    marker_colors=colors,
    textinfo='label+percent',
    textposition='inside'
)])

# Update layout
fig.update_layout(
    title="Cost Savings Potential: $430m Annual",
    uniformtext_minsize=14, 
    uniformtext_mode='hide',
    showlegend=True,
    legend=dict(
        orientation='v',
        yanchor='middle',
        y=0.5,
        xanchor='left',
        x=1.01
    )
)

# Add center text showing total cost
fig.add_annotation(
    text="$430m<br>Total Cost",
    x=0.5, y=0.5,
    font_size=20,
    showarrow=False
)

# Update traces
fig.update_traces(
    hovertemplate="<b>%{label}</b><br>" +
                  "Amount: $%{value:,.0f}<br>" +
                  "Percentage: %{percent}<br>" +
                  "<extra></extra>",
    textfont_size=14
)

# Save the chart
fig.write_image("cost_savings_donut.png")