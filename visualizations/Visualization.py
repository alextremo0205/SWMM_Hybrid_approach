import io
from PIL import Image
import plotly.io as pio

import pandas as pd
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
        width=1600,
        height=800,
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



if __name__ == '__main__':
    history = {
        'Training loss':
            {0:1,
            1:2,
            2:3,
            3:4
            },
        'Validation loss':
            {0:2,
            1:3,
            2:4,
            3:5}
    }
    
    
    fig = go.Figure()
    
    scatter_loss     = get_scatter_from_dict(history, 'Training loss')
    scatter_val_loss = get_scatter_from_dict(history, 'Validation loss')
    
    fig.add_trace(scatter_loss)
    fig.add_trace(scatter_val_loss)
    
    fig = style_loss_fig(fig)
    
    show(fig)

































# import pandas as pd
# from matplotlib import pyplot as plt

# def plot_loss(history):
#     fig, ax = plt.subplots(nrows = 1, ncols = 1)
#     df = pd.DataFrame(history)

#     fig, ax = plt.subplots()
#     plt.style.use('seaborn')
#     ax.plot(df['Loss'])
    
#     return fig, ax


# if __name__ == '__main__':

    
#     fig, ax = plot_loss(history)
#     plt.show()
    
# # ax.set_xlabel('Epochs')
# # ax.set_ylabel('MSE Loss')
# # ax.set_title('')

# # plt.tight_layout
# # plt.show()