import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from vae import VAE

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
# from torchsummary import summary


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


def train_vae(dataset, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae = VAE().to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    bs = 32

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    for epoch in range(epochs):
        for idx, (images, _) in enumerate(dataloader):
            recon_images, mu, logvar = vae(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch + 1,
                                                                        epochs, loss.data / bs, bce.data / bs,
                                                                        kld.data / bs)
            print(to_print)

    torch.save(vae.state_dict(), 'vae.torch')
    return vae


def load_vae(fname="vae.torch"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE().to(device)
    vae.load_state_dict(torch.load(fname, map_location='cpu'))
    return vae


def compare(x, vae):
    recon_x, _, _ = vae(x)
    return torch.cat([x, recon_x])


def main():
    dataset = datasets.ImageFolder(root='./shape_images', transform=transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
    ]))

    vae = train_vae(dataset, epochs=15)
    # vae = load_vae()

    fixed_x = dataset[1][0].unsqueeze(0)
    compare_x = compare(fixed_x, vae)

    save_image(compare_x.data.cpu(), 'sample_image.png')


if __name__ == "__main__":
    main()
