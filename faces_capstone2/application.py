import base64
from io import BytesIO
from PIL import Image

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

from keras.models import load_model, Model, Sequential
from keras.layers import Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.embeddings import Embedding
from keras import backend as K
K.set_image_data_format('channels_last')
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

MODEL_FN = 'decoder_25c.h5'
BATCH_SIZE = 32
N_COMPONENTS = 25
N_IMAGES = 200

X = np.arange(N_IMAGES)
X = np.array(X)

model = load_model(MODEL_FN)
decoder = K.function([model.get_layer('decoder_input').input, K.learning_phase()], model.layers[-1].output)
encoder = Model(inputs=model.input, outputs=model.get_layer('encoder_output').output)

encoded = encoder.predict(X, batch_size=BATCH_SIZE)
component_mean = np.mean(encoded, axis=0)
component_std = np.std(encoded, axis=0)
covariance = np.cov((encoded - component_mean).T)
eigvals, eigvecs = np.linalg.eig(covariance)
idx = np.argsort(-eigvals)
eigvals = np.sqrt(eigvals[idx])
eigvecs = eigvecs[:, idx]

def numpy_to_b64(array):
    pil = Image.fromarray(array)
    bytes = BytesIO()
    im.save(bytes, format='png')
    b64_img = base64.b64encode(bytes.getvalue()).decode("utf-8")
    return b64_img

FACE_WIDTH = '30%'
N_SLIDER_COLS = 1
N_SLIDER_ROWS = 25
SLIDER_WIDTH = '20%'
#SLIDER_HEIGHT = W/E
SLIDER_MIN = -3.0
SLIDER_MAX = 3.0
INITIAL_VAL = 0
SLIDER_STEP = 0.01
colors = {'background': '#111111', 'text': '#7FDBFF'}
FACE_STYLE = {'width':FACE_WIDTH, 'display':'inline-block', 'margin-left':'auto', 'margin-right':'auto'}
COLUMN_STYLE = {'width':SLIDER_WIDTH, 'display':'inline-block', 'margin-left':20}
TEXT_STYLE = {'font-family':'sans-serif', 'color':colors['text'], 'text-align':'center', 'margin':'auto'}

def numpy_to_b64(array):
    pil = Image.fromarray(array)
    bytes = BytesIO()
    pil.save(bytes, format='png')
    b64_img = base64.b64encode(bytes.getvalue()).decode("utf-8")
    return b64_img

app = dash.Dash()
app.config.suppress_callback_exceptions = True

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Face Generator',
        style=TEXT_STYLE),
    html.Div(children=' ', style={
        'textAlign': 'center',
        'color': colors['text']}),
    (
    #where face and controls go
    html.Div([
        html.Div([
            html.Div(dcc.RadioItems(id='radio', options=[{'label':'Randomize', 'value':'randomize'}, 
                    {'label':'Reset', 'value':'reset'}], value='randomize'), style=TEXT_STYLE),
            html.Div(html.Button('Submit', id='submit'), style=TEXT_STYLE),
            html.Div(html.Img(id='face'), style=FACE_STYLE),
            ], style={'display':'inline-block', 'margin':'auto', 'width':FACE_WIDTH}),
        html.Div(id='sliders', style={'width':'70%', 'display':'inline-block', 'margin':'auto'})]
    )
    )
    ])

@app.callback(Output('sliders', 'children'), [Input('sliders', 'id')])
def create_sliders(id):
    slider_cols = [html.Div('Principal Components', style=TEXT_STYLE)]
    for col in range(N_SLIDER_COLS):
        slider_rows = []
        for row in range(N_SLIDER_ROWS):
            slider_id = N_SLIDER_ROWS * col + (row + 1)
            slider = dcc.Slider(id='component-{}'.format(slider_id), min=SLIDER_MIN, max=SLIDER_MAX, step=SLIDER_STEP, value=INITIAL_VAL)
            slider_rows.append(slider)
        columns = html.Div(id='column-{}'.format(col+1), children=slider_rows, style=COLUMN_STYLE)
        slider_cols.append(columns)
    return slider_cols

input_sliders = []
state_sliders = []
output_sliders = []
slider_vals = {}
for i in range(N_COMPONENTS):
    slider_id = 'component-{}'.format(i+1)
    slider_vals[slider_id] = INITIAL_VAL
    input_sliders.append(Input(slider_id, 'value'))
    state_sliders.append(State(slider_id, 'value'))
    output_sliders.append(Output(slider_id, 'value'))

@app.callback((Output('face', 'src')), input_sliders)
def get_new_face(*args):
    ctx = dash.callback_context
    try:
        if ctx.triggered:
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            triggered_val = ctx.triggered[0]['value']
            slider_vals[triggered_id] = triggered_val
    except:
        low_dim = np.expand_dims(component_mean, axis=0)
        default_face = decoder([low_dim, 0])[0]
        default_face = (default_face * 255.0).astype(np.uint8)
        figure = numpy_to_b64(default_face)
        src = 'data:image/png;base64,{}'.format(figure)
    #low_dim = component_mean + np.dot(eigvecs, (list(slider_vals.values()) * eigvals).T).T
    low_dim = component_mean + list(slider_vals.values()) * component_std
    low_dim = np.expand_dims(low_dim, axis=0)
    pred_face = decoder([low_dim, 0])[0]
    new_face = (pred_face * 255.0).astype(np.uint8)
    figure = numpy_to_b64(new_face)
    src = 'data:image/png;base64,{}'.format(figure)
    return src

@app.callback(output_sliders, [Input('submit', 'n_clicks')], [State('radio', 'value')])
def random_reset(n_clicks, value):
    if n_clicks > 0:
        if value == 'randomize':
            return list(np.clip(np.random.randn(N_COMPONENTS), -3.0, 3.0))
        elif value == 'reset':
            return [0]*N_COMPONENTS
        
if __name__ == '__main__':
    app.run_server(debug=True)
