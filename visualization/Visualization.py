import io
from PIL import Image
import plotly.io as pio

import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def plot_heads_timeseries(swmm_heads_pd, predicted_heads_pd, runoff_pd, node):

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    scatter_swmm = get_scatter(swmm_heads_pd, node, 'SWMM')
    scatter_pred = get_scatter(predicted_heads_pd, node, 'GNN')
    bar_runoff   = get_barplot(runoff_pd, node, 'Runoff')
    
    
    fig.add_trace(bar_runoff,   secondary_y=True)
    
    fig.add_trace(scatter_swmm, secondary_y=False)
    fig.add_trace(scatter_pred, secondary_y=False)
    # fig.update_yaxes(autorange="reversed",  secondary_y=True)
    fig.update_yaxes(range=[0.03,  0],  secondary_y=True)
    
    fig = style_heads_fig(fig, node)
    
    fig.show()

    
def plot_loss(history):
    fig = go.Figure()
    
    scatter_loss     = get_scatter(history, 'Training loss')
    scatter_val_loss = get_scatter(history, 'Validation loss')
    
    fig.add_trace(scatter_loss)
    fig.add_trace(scatter_val_loss)
    
    fig = style_loss_fig(fig)
    
    fig.show()
    
def get_scatter(df, field, name = None):

    if name == None:
        name = field
    
    if type(df) == dict:
        df = pd.DataFrame(df)
        
    df = df.round(2)
    
    scatter_trace = go.Scatter(x = df.index.values, 
                               y = df[field], 
                               
                               name = name,
                               mode = "lines+markers",

                               line = dict(width = 3),
                               marker=dict(size = 8),
                               
                               )
    
    
    return scatter_trace
   

def get_barplot(df, node, name):
    df = pd.DataFrame(df[node].reset_index(drop = True)).round(2)

    if name == None:
        name = node

    bar_trace = go.Bar( x = df.index.values, 
                        y = df[node], 
                        # name = name,
                        # mode = "lines+markers",
                        # line = dict(width = 3),
                        # marker=dict(size = 8),
                        )
    return bar_trace

def style_loss_fig(fig):
    margin = 80
    
    fig.update_layout(
        width=900,
        height=600,
        margin=dict(l=margin, r=margin, t=margin, b=margin),
        
        title       ="Loss function",
        xaxis_title ="Epochs",
        yaxis_title ="MSE Loss",
        legend_title="Legend",
        
        # font=dict(family="Times new Roman", size=28, color="Black"),
        template = custom_template, 

    )

    return fig

def style_heads_fig(fig, node):
    margin = 80
    
    fig.update_layout(
        width=900,
        height=600,
        margin=dict(l=margin, r=margin, t=margin, b=margin),
        
        title       ="Head timeseries at node " + node,
        xaxis_title ="Time steps (5 min)",
        yaxis_title ="Head (MSL)",
        legend_title="Legend",
        
        template = custom_template, 

    )

    return fig


def show(fig):

    buf = io.BytesIO()
    pio.write_image(fig, buf)
    img = Image.open(buf)
    img.show() 

def animate_nodal_depth(df):
    net = px.scatter(
        df, 
        x="x_coord", 
        y="y_coord", 
        size="Depth", 
        animation_frame="Time", 
        size_max=20, 
        hover_name="Node",
        width=500, 
        height=800
        )
    return net


def plot_nodal_variable(value, ref_window, name_value, colorscale='PuBu', ref_marker_size = 5):
    
    num_steps = value.shape[1]
    
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    
    fig_dict["layout"]["updatemenus"] = [
    {
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 500, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 300,
                                                                    "easing": "quadratic-in-out"}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }
    ]
    
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time step (5 min): ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    
    for time_step in range(1,num_steps):
        fig_dict['frames'].append(go.Frame(
            data=get_bubble_trace(value[:,time_step], ref_window, colorscale = colorscale, ref_marker_size = ref_marker_size),
            name = time_step
            )
                                  )
        
        slider_step = {"args": [
            [time_step],
            {"frame": {"duration": 300, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 300}}
        ],
            "label": time_step,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)
        
    fig_dict["layout"]["sliders"] = [sliders_dict]
    
    
    data = get_bubble_trace(value[:,0], ref_window, colorscale = colorscale, ref_marker_size= ref_marker_size)
    
    fig = go.Figure(data =data, layout = fig_dict['layout'], frames = fig_dict["frames"])
    
    fig = style_bubble_fig(fig, name_value)
    fig.show()
    
    return fig

def get_bubble_trace(value, ref_window, colorscale, ref_marker_size=5):
    node_names = ref_window.name_nodes
    coordinates = ref_window.pos.numpy()
    x_coord = coordinates[:,0]
    y_coord = coordinates[:,1]

    if type(value)==torch.Tensor:
        value = value.numpy()
    
    sizeref = 2. * max(value) / (ref_marker_size ** 2)
    scatter_trace= go.Scatter(x=x_coord, y=y_coord,
                    mode='markers',
                    name='coordinates',
                    hovertemplate = '%{text}', 
                    text =  ['<b><br> Node ID: </b> {name} <br> <b>Value:</b> {value:.2f}'.format(name = node_names[i], value = value[i]) for i in range(len(node_names))],
                    marker_size=value-min(value),
                    marker=dict(color=value,showscale=True, sizeref=sizeref, sizemin = 2, colorscale=colorscale, cmax=1, cmin =0, 
                                line=dict(width=1,color='DarkSlateGrey')),
                    )
    return scatter_trace    
    

def style_bubble_fig(fig, name_value):
    fig.update_layout(
        width=500,
        height=800,
        title       ="Spatial distribution of "+ name_value, 
        legend_title="Legend",
        xaxis = dict(visible=False),
        yaxis = dict(visible=False),
        font=dict(family="Times new Roman", size=18, color="Black"),
    )
    return fig



custom_template = {
    "layout": go.Layout(
        font={
            "family": "Nunito",
            "size": 16,
            "color": "#707070",
        },
        title={
            "font": {
                "family": "Lato",
                "size": 22,
                "color": "#1f1f1f",
            },
        },
        xaxis={
            "showspikes":   True,
            "spikemode":    'across',
            "spikesnap":    'cursor',
            "showline":     True,
            "showgrid":     True,
        },
        hovermode  = 'x',
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        colorway=px.colors.qualitative.G10,
    )
}

