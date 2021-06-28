import torch
import torch.nn as nn

BCE_loss = nn.BCELoss()
L1_loss = nn.L1Loss()

def G_loss(proba_matrix, generated_B, real_B, lam=0.5):
  '''
  Generator loss
  params:
      proba_matrix - discriminator output
      generated_B - generated image B
      real_B - real image B
  '''
  bce = BCE_loss(proba_matrix, torch.ones_like(proba_matrix))
  l1 = L1_loss(generated_B, real_B)
  return bce + lam*l1

def D_loss(proba_matrix, real):
  '''
  Discriminator loss
  Params:
      proba_matrix - discriminator output
      real (boolean) - true : real image, false : generated image
  '''
  if real == True: real_matrix = torch.ones_like(proba_matrix)
  else: real_matrix = torch.zeros_like(proba_matrix)

  return BCE_loss(proba_matrix, real_matrix)

