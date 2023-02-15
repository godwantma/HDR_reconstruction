import torch
import cv2
from model import Model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import math
from skimage.metrics import structural_similarity

def itm(model, input_image, tau):
    input_image = input_image / 255
    input_image = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0)
    max_c = input_image[0].max(dim=0).values - tau
    max_c[max_c < 0] = 0
    alpha = (max_c / (1 - tau)).float()

    with torch.no_grad():
        output_image = model(input_image)

    output_image = (((1 - alpha) * input_image + alpha * output_image)**3).squeeze().permute(1, 2, 0).detach()
    return output_image


if __name__ == '__main__':
    # Load model
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = Model(device).to(device)
    model.eval()
    model.load_state_dict(torch.load('output_model3/weight.pth', map_location=device))

    psnr_path = open('test/results/psnr.txt','w+')


    # Load images
    for i in range(1, 8, 1):
        ldr = cv2.imread('test/' + str(i) + '.png')
        ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB, cv2.IMREAD_ANYDEPTH)
        ldr = ldr

        hdr = cv2.imread('test/' + str(i) + '.hdr', cv2.IMREAD_ANYDEPTH)
        hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)

        ldr = torch.tensor(ldr)

        ldr = ldr.to(torch.device('cuda'))

        hdr_reconstructed = itm(model, ldr, tau=0.8)

        ldr = ldr.cpu().data.numpy()
        hdr_reconstructed = hdr_reconstructed.cpu().data.numpy()
        mse = np.mean((hdr - hdr_reconstructed) ** 2)
        psnr = 10 * math.log10(1.0 ** 2 / mse)
        psnr_path.write('psnr' + str(i) + ':' + str(psnr) + '\n')
        ssim = structural_similarity(hdr, hdr_reconstructed, data_range=1.0, multichannel=True)
        psnr_path.write('ssim' + str(i) + ':' + str(ssim) + '\n')

        hdr_ETM_reconstructed = hdr_reconstructed ** (1 / 3)
        hdr_ETM = hdr ** (1 / 3)

        hdr_reconstructed = np.log(1. + 2000. * hdr_reconstructed) / np.log(1. + 2000.)
        hdr = np.log(1. + 2000. * hdr) / np.log(1. + 2000.)

        # Save result
        fig = plt.figure(figsize=(15, 3))

        fig.add_subplot(1, 5, 1)
        plt.imshow(ldr)
        plt.axis('off')
        plt.title('Input')

        fig.add_subplot(1, 5, 2)
        plt.imshow(hdr_reconstructed)
        plt.axis('off')
        plt.title('Reconstruction by LTM')

        fig.add_subplot(1, 5, 3)
        # plt.imshow(hdr ** (1 / 3))
        plt.imshow(hdr)
        plt.axis('off')
        plt.title('Ground truth by LTM')

        fig.add_subplot(1, 5, 4)
        plt.imshow(hdr_ETM_reconstructed)
        plt.axis('off')
        plt.title('Reconstruction by GTM')

        fig.add_subplot(1, 5, 5)
        plt.imshow(hdr_ETM)
        plt.axis('off')
        plt.title('Ground truth by GTM')

        plt.savefig('test/results/' + str(i) + '_test.png', dpi=500, bbox_inches='tight')

        np.save('test/results/' + str(i) + '_hdrrec.npy', hdr_reconstructed)

    psnr_path.close()
