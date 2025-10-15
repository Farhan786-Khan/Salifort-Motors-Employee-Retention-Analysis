import plotly.graph_objects as go
import json

# Parse the JSON data
data = [
  {
    "chart_type": "turnover_by_projects",
    "data": [
      {"number_project": 2, "turnover_rate": 56.1, "stayed": 43.9},
      {"number_project": 3, "turnover_rate": 46.5, "stayed": 53.5},
      {"number_project": 4, "turnover_rate": 46.7, "stayed": 53.3},
      {"number_project": 5, "turnover_rate": 65.9, "stayed": 34.1},
      {"number_project": 6, "turnover_rate": 65.5, "stayed": 34.5},
      {"number_project": 7, "turnover_rate": 99.3, "stayed": 0.7}
    ]
  }
]

# Extract data for the turnover by projects chart
turnover_data = data[0]["data"]
projects = [item["number_project"] for item in turnover_data]
turnover_rates = [item["turnover_rate"] for item in turnover_data]

# Create bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=projects,
    y=turnover_rates,
    marker_color='#1FB8CD',
    width=0.6,  # Add spacing between bars
    text=[f'{rate}%' for rate in turnover_rates],  # Add data labels
    textposition='outside',
    hovertemplate='Projects: %{x}<br>Turnover Rate: %{y}%<extra></extra>'
))

# Update layout
fig.update_layout(
    title='Employee Turnover by Project Count',
    xaxis_title='Number of Projects',  # More professional label
    yaxis_title='Turnover Rate (%)'
)

# Update traces and axes according to guidelines
fig.update_traces(cliponaxis=False)
fig.update_yaxes(range=[0, 100])  # Limit to 0-100%
fig.update_xaxes(tickmode='linear', dtick=1)

# Save the chart
fig.write_image('turnover_by_projects.png')
