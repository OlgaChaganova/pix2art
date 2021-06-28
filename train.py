import torch
import numpy as np
import matplotlib.pylab as pl
from IPython import display
from tqdm.notebook import tqdm

from models.discriminator import Discriminator
from models.generator import Generator
from utils import config
from utils.visualise import show_images
from utils.weights_init import weights_init
from loss_functions import  D_loss, G_loss
from data.dataloaders import get_dataloader


def fit(discriminator, generator, d_loss, g_loss, d_optim, g_optim, train_dl, test_dl,
        epochs, LAM=2, start_idx=1):
    discriminator.train()
    generator.train()
    torch.cuda.empty_cache()

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    for epoch in range(epochs):
        loss_d_per_epoch = []
        loss_g_per_epoch = []
        real_score_per_epoch = []
        fake_score_per_epoch = []

        for A_images, B_images in tqdm(train_dl):
            B_images = B_images.to(config.DEVICE)
            A_images = A_images.to(config.DEVICE)
            batch_size = B_images.shape[0]

            # -----------1 Train discriminator--------------------
            d_optim.zero_grad()

            # 1-1 Пропускаем настоящие изображения через дискриминатор
            real_AB = torch.cat([A_images, B_images], dim=1)
            real_preds = discriminator(real_AB)  # вероятности для патчей
            d_real_loss = d_loss(real_preds, real=True)  # учим дискриминатор предсказывать реальные изображения
            cur_real_score = torch.mean(real_preds).item()

            # 1-2 Пропускаем сгенерированные изображения через дискриминатор
            fake_images = generator(A_images)
            fake_AB = torch.cat([A_images, fake_images.detach()], dim=1)
            fake_preds = discriminator(fake_AB)  # вероятности для патчей
            d_fake_loss = d_loss(fake_preds, real=False)  # учим дискриминатор предсказывать сгенерированные изображения
            cur_fake_score = torch.mean(fake_preds).item()

            real_score_per_epoch.append(cur_real_score)
            fake_score_per_epoch.append(cur_fake_score)

            # 1-3 Обновляем веса
            loss_d = (d_real_loss + d_fake_loss) / 2  # согласно статье делим на 2
            loss_d.backward()
            d_optim.step()

            loss_d_per_epoch.append(loss_d.item())

            # ------------2 Обучаем генератор----------------
            generator.zero_grad()

            # 2-1 Предсказания для фейковых картинок
            fake_preds = discriminator(fake_AB)
            loss_g = g_loss(fake_preds, fake_images, B_images, LAM)  # bce + L1

            # 2-2 Обновляем веса
            loss_g.backward()
            g_optim.step()

            loss_g_per_epoch.append(loss_g.item())

        # Сохраняем статистику
        losses_g.append(np.mean(loss_g_per_epoch))
        losses_d.append(np.mean(loss_d_per_epoch))
        real_scores.append(np.mean(real_score_per_epoch))
        fake_scores.append(np.mean(fake_score_per_epoch))

        # Progress bar
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch + 1, epochs,
            losses_g[-1], losses_d[-1], real_scores[-1], fake_scores[-1]))

        # Каждые 6 эпох показываем генерацию на валидационном сете
        if (epoch + 1) % 6 == 0:
            print('\t\t\t\t\tVALIDATION\n')
            A_images, B_images = next(iter(test_dl))
            fake_images = generator(A_images.to(config.DEVICE))
            show_images(A_images.cpu(), B_images.cpu(),
                        fake_images.cpu(), batch_size)
            display.display(pl.gcf())


        # Каждые 3 эпохи показываем генерацию на тренировочном сете
        elif (epoch + 1) % 3 == 0:
            print('\t\t\t\t\tTRAIN\n')
            show_images(A_images.cpu(), B_images.cpu(),
                        fake_images.cpu(), batch_size)
            display.display(pl.gcf())

        # Save model's weights every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save({
                'G_state_dict': generator.state_dict(),
                'G_optimizer_state_dict': g_optim.state_dict(),
                'G_loss': loss_g,
            }, config.PATH_TO_SAVE_MODEL + 'Pix2Pix_Generator-{}'.format(start_idx + epoch))

            torch.save({
                'D_state_dict': discriminator.state_dict(),
                'D_optimizer_state_dict': d_optim.state_dict(),
                'D_loss': loss_d,
            }, config.PATH_TO_SAVE_MODEL + 'Pix2Pix_Discriminator-{}'.format(start_idx + epoch))

    return losses_g, losses_d, real_scores, fake_scores

def main():
    D_net = Discriminator().to(config.DEVICE)
    G_net = Generator().to(config.DEVICE)

    G_net.apply(weights_init)
    D_net.apply(weights_init)

    G_optimizer = torch.optim.Adam(G_net.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    D_optimizer = torch.optim.Adam(D_net.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))

    train_loader = get_dataloader('train', 2)
    val_loader = get_dataloader('val')

    print(config.DEVICE)

    history = fit(D_net, G_net,
                  D_loss, G_loss,
                  D_optimizer, G_optimizer,
                  train_loader, val_loader, 100, LAM=100, start_idx=1)


if __name__ == "__main__":
    main()
