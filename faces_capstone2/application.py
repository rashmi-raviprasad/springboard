import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from keras.models import load_model, Model, Sequential
from keras.layers import Activation, Input, Flatten, Reshape
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.embeddings import Embedding
from keras import backend as K
K.set_image_data_format('channels_last')
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

MODEL_FN = 'decoder_test.h5'
BATCH_SIZE = 32
N_COMPONENTS = 100
N_IMAGES = 500

X = np.arange(N_IMAGES)
X = np.array(X)

model = load_model(MODEL_FN)
decoder = K.function(model.get_layer('reshape_1').input, model.layers[-1].output)
encoder = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)

encoded = encoder.predict(X, batch_size=BATCH_SIZE)
component_mean = np.mean(encoded, axis=0)
component_std = np.std(encoded, axis=0)
covariance = np.cov((encoded - component_mean).T)
eigvals, eigvecs = np.linalg.eig(covariance)
idx = np.argsort(-eigvals)
eigvals = np.sqrt(eigvals[idx])
eigvecs = eigvecs[:, idx]

def get_new_face(new_params):
    low_dim = component_mean + np.dot(eigvecs, (new_params * eigvals).T).T
    low_dim = np.expand_dims(low_dim, axis=0)
    pred_face = decoder(low_dim)[0]
    new_face = (pred_face * 255.0).astype(np.uint8)
    return new_face

app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(children='Dash: A web application framework for Python.', style={
        'textAlign': 'center',
        'color': colors['text']
    })
])


if __name__ == '__main__':
    app.run_server(debug=True)
