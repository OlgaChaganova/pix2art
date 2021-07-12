import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH_TO_SAVE_MODEL = '//trained_model'

PATH_TO_DATA = 'C:/Users/olgac/PycharmProjects/pix2art/pix2pix dataset/facades'

PATH_AYVAZOVSKY = 'https://github.com/OlgaChaganova/pix2art/releases/download/v2.0/Ayvazovsky_Pix2Pix_Generator-250.pth'
PATH_REMBRANDT = 'https://github.com/OlgaChaganova/pix2art/releases/download/v2.0/Rembrandt_Pix2Pix_Generator-250.pth'

PATH_EXAMPLES_AYVAZOVSKY = 'https://raw.githubusercontent.com/OlgaChaganova/pix2art/main/pix2art_streamlit/test%20images%20ayvazovsky/'
PATH_EXAMPLES_REMBRANDT = 'https://raw.githubusercontent.com/OlgaChaganova/pix2art/main/pix2art_streamlit/test%20images%20rembrandt/'

TEST_IMAGES_AYVAZOVSKY = [174, 53, 377, 23, 85, 91, 305, 337, 424]
TEST_IMAGES_REMBRANDT = [114, 115, 218, 26, 70, 274, 2, 229, 126]

# Adam parameters
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
