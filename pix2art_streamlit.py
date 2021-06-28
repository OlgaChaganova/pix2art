import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
from PIL import Image
import requests
import numpy as np

from models.generator import Generator
from utils import config


@st.cache
def load_model(style_type):
    model = Generator().to(config.DEVICE)
    if style_type == 'Ayvazovsky':
        PATH = config.PATH_AYVAZOVSKY
    else:
        PATH = config.PATH_REMBRANDT

    state_dict = torch.hub.load_state_dict_from_url(PATH, map_location=config.DEVICE)
    model.load_state_dict(state_dict['G_state_dict'])
    return model


@st.cache
def denormalize(inp):
    inp = inp.numpy().transpose((1, 2, 0))  # транспонирует тензор
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # убираем нормализацию изображения (возвращаем в исходное состояние)
    inp = np.clip(inp, 0, 1)
    return inp


@st.cache
def transform_input(image):
    '''
    :param image: type - np.array
    :return: torch.tensor [1, 3, 256, 256]
    '''
    transform = transforms.Compose([
        transforms.Resize(size=(286, 286)),
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tr_img = transform(image)
    return tr_img.unsqueeze(0)


def use_example(TEST_IMAGES, PATH_EXAMPLES):
    rand_ind = np.random.choice(TEST_IMAGES)
    user_image = Image.open(requests.get(PATH_EXAMPLES + str(rand_ind) + '.jpg', stream=True).raw)
    return user_image


def transform_toPIL(image):
    transform = transforms.ToPILImage()
    return transform(image)


def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    # Sidebar
    style = st.sidebar.selectbox("Style:", ("Ayvazovsky", "Rembrandt"))
    st.sidebar.write('I will...')
    mode = st.sidebar.radio('', ("draw image here", "upload image", 'use example image'))

    if mode == "draw image here":
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 8, 1) # ширина линии
        drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"))

    # Main page
    st.title('Pix2Art.  Ayvazovsky & Rembrandt Edition')

    st.subheader('With love to Machine Learning and Fine Art <3')

    st.write('What to do:\n'
            '   * imagine that you are a great artist; \n'
            '   * draw something in the canvas below or upload sketch via loader;\n'
            '   * click *Draw!* and see how you drawing turns into pure art.')


    if mode == 'draw image here':
        # Create a canvas component
        st.write('Draw a sketch:')
        canvas_draw = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color='#ffffff', #ffffff
            background_color='#000000', #000000
            height=256,
            width=256,
            drawing_mode=drawing_mode,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            key="full_app",
        )
        if canvas_draw.json_data is not None:
            st.subheader('Great! Are you ready to see the magic of art?')

    elif mode == 'upload image':
        # Upload image
        uploaded_img = st.file_uploader("Choose a file", ["png", "jpg", "jpeg"])
        if uploaded_img is not None:
            st.write('Got it! Here is your sketch:')
            show_file = st.empty()
            show_file.image(uploaded_img)
            user_image = load_image(uploaded_img)
            st.subheader('Are you ready to see the magic of art?')


    elif mode == 'use example image':
        if style == 'Ayvazovsky':
            PATH_EXAMPLES = config.PATH_EXAMPLES_AYVAZOVSKY
            TEST_IMAGES = config.TEST_IMAGES_AYVAZOVSKY
        elif style == 'Rembrandt':
            PATH_EXAMPLES = config.PATH_EXAMPLES_REMBRANDT
            TEST_IMAGES = config.TEST_IMAGES_REMBRANDT

        user_image = use_example(TEST_IMAGES, PATH_EXAMPLES)
        st.write('Here is a sketch:')
        example_image = st.image(user_image)

        st.subheader('Are you ready to see the magic of art?')


    st.write('\n')
    draw = st.button('Draw!')

    if draw:
        st.write('Wait a sec...')

        if mode == 'draw image here':
            user_image = canvas_draw.image_data
            user_image = user_image[:, :, :-1]  # need to remove 4th channel
            user_image = Image.fromarray(user_image.astype(np.uint8))

        user_image = transform_input(user_image)
        user_image = torch.cat([user_image, user_image], dim=0).to(config.DEVICE) # single image cannot go through dropout layer
        G_net = load_model(style)

        output_image = G_net(user_image.float())

        # st.write(output_image)
        output_image = denormalize(output_image[0].detach())

        # visualise picture
        st.image(output_image, clamp=True, channels='RGB')
        st.write('Here you are! How is it?')


main()

