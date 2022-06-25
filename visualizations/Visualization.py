import io
from PIL import Image
import plotly.io as pio

import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


def plot_loss(history):
    fig = go.Figure()
    
    scatter_loss     = get_scatter_from_dict(history, 'Training loss')
    scatter_val_loss = get_scatter_from_dict(history, 'Validation loss')
    
    fig.add_trace(scatter_loss)
    fig.add_trace(scatter_val_loss)
    
    fig = style_loss_fig(fig)
    
    fig.show()

def get_scatter_from_dict(history, field):
    df = pd.DataFrame(history)
    
    scatter_trace = go.Scatter(x = df.index.values, 
                               y = df[field], 
                               
                               name = field,
                               mode = "lines+markers",

                               line = dict(width = 3),
                               marker=dict(size = 12),
                               
                               )
    
    
    return scatter_trace
   
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
        
        font=dict(family="Times new Roman", size=28, color="Black"),

    )

    return fig

def show(fig):

    buf = io.BytesIO()
    pio.write_image(fig, buf)
    img = Image.open(buf)
    img.show() 




# def get_depths_to_rows(self,timestep):
    # rows = [[timestep, node, (depth - self.original_min[node]).item(), self.pos[node][0], self.pos[node][1]] for node, depth in self.h.items()]
#     return rows

columns = ['Time' , 'Node', 'Depth', 'x_coord' , 'y_coord']

# df = pd.DataFrame(wn.get_depths_to_rows(0), columns = columns)

# for time in range(1):
#     wn.update_h(runoff = X_train[0][2][time]) 
#     new_h_rows = pd.DataFrame(wn.get_depths_to_rows(time+1), columns = columns)
#     df = pd.concat([df,new_h_rows])


# depth_one_node = df[df['Node']=='j_90376']['Depth'] #.plot() j_90550
# depth_one_node=depth_one_node.reset_index()
# depth_one_node['Depth'].plot()


# net = utils.animate_nodal_depth(df)




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
    dt = 5
    
    # make figure
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
                    hovertemplate = 'Value: %{text:.2f}<extra></extra>',
                    text = value,
                    marker_size=value-min(value),
                    marker=dict(color=value,showscale=True, sizeref=sizeref, sizemin = 2, colorscale=colorscale,  
                                line=dict(width=2,color='DarkSlateGrey')),
                    
                    )
    return scatter_trace    
    

def style_bubble_fig(fig, name_value):
    margin = 80
    
    fig.update_layout(
        width=500,
        height=800,
        # margin=dict(l=margin, r=margin, t=margin, b=margin),
        
        title       ="Spatial distribution of "+ name_value, 
        # xaxis_title ="Epochs",
        # yaxis_title ="MSE Loss",
        legend_title="Legend",
        xaxis = dict(visible=False),
        yaxis = dict(visible=False),
        
        font=dict(family="Times new Roman", size=18, color="Black"),
        # updatemenus=[dict(
        #     type="buttons",
        #     buttons=[dict(label="Play2",
        #                   method="animate",
        #                   args=[None])])]
        
        


    )

    return fig