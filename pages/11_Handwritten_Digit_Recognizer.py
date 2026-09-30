import numpy as np
import streamlit as st
from PIL import Image
import keras
from keras.layers import Dense
from streamlit_drawable_canvas import st_canvas

# ---------- Page config ----------
st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✍️", layout="centered")


# ---------- Data loading (cached so it only runs once) ----------
@st.cache_data(show_spinner="Loading and preprocessing MNIST dataset...")
def load_data():
    (xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()

    xtrain = xtrain / 255.0
    xtest = xtest / 255.0

    xtrain = xtrain.reshape(-1, 784)
    xtest = xtest.reshape(-1, 784)

    return xtrain, ytrain, xtest, ytest


@st.cache_resource(show_spinner="Training model... (this happens once)")
def train_model(xtrain, ytrain):
    model = keras.Sequential()
    model.add(Dense(64, activation='relu', input_dim=784))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(xtrain, ytrain, epochs=5, verbose=0)
    return model


def preprocess_canvas_image(image_data):
    # Convert the RGBA canvas array to grayscale, resize to 28x28 to match training data
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA').convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 784)


# ---------- UI ----------
st.title("✍️ Handwritten Digit Recognizer")
st.write(
    "Draw a single digit (0-9) on the canvas below, and a neural network trained on "
    "the MNIST dataset will predict what you drew."
)

xtrain, ytrain, xtest, ytest = load_data()
model = train_model(xtrain, ytrain)

st.subheader("Draw a digit")
col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        return_image_data=True,
    )

with col2:
    if canvas_result.image_data is not None:
        preview = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L').resize((28, 28))
        st.write("Model input preview (28×28):")
        st.image(preview, width=140)

if st.button("Predict Digit", type="primary"):
    if canvas_result.image_data is None or canvas_result.image_data[:, :, :3].sum() == 0:
        st.warning("Please draw a digit on the canvas first.")
    else:
        input_data = preprocess_canvas_image(canvas_result.image_data)
        prediction = model.predict(input_data, verbose=0)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        st.success(f"### Predicted Digit: {predicted_digit}")
        st.write(f"Confidence: **{confidence:.1f}%**")
        st.bar_chart(prediction[0])

st.divider()
st.caption("Built with Streamlit • Keras • Data: MNIST Handwritten Digits Dataset")