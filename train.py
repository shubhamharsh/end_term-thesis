import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import dnnlib
import legacy
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--gpu", action="store", dest="gpu", help="separate numbers with commas, eg. 3,4,5", required=True)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpus = args.gpu.split(",")
n_gpu = len(gpus)


# ------------------ Encoder using ResNet-34 as the backbone ------------------

class EncoderResNet34(nn.Module):
    def __init__(self, latent_dim=512):
        super(EncoderResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove final fully connected layer
        self.resnet = self.resnet.cuda()

        # Calculate the flattened size of ResNet output
        dummy_input = torch.randn(1, 3, 256, 256).cuda()
        with torch.no_grad():
            dummy_output = self.resnet(dummy_input)
        self.fc_input_size = dummy_output.view(1, -1).size(1)

        # Fully connected layer to map features to latent space (W space)
        self.fc = nn.Linear(self.fc_input_size, latent_dim).cuda()

    def forward(self, x):
        x = x.cuda()
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        w = self.fc(x)
        return w

# ------------------ StyleGAN2 Model Loader ------------------

def load_stylegan2_model(model_path):
    with open(model_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].cuda()  # 'G_ema' is the generator part of StyleGAN2
    return G

# ------------------ Discriminator (Simple Binary Classifier) ------------------

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ------------------ Dataset and DataLoader ------------------

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Data transformations for the images (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# DataLoader for HQ images
hq_train_loader = DataLoader(ImageDataset("datasets/HQ_Train", transform), batch_size=16, shuffle=True)
lq_train_loader = DataLoader(ImageDataset("datasets/LQ_Train", transform), batch_size=16, shuffle=True)

# ------------------ Initialize Models ------------------

# Initialize the encoder and StyleGAN2 generator
encoder_hq = EncoderResNet34().cuda()
encoder_lq = EncoderResNet34().cuda()
stylegan_generator = load_stylegan2_model(r'/home/cvblhcs/Desktop/sharsh_thesis/ffhq.pkl').eval()  # Path to your pretrained model

# Initialize the discriminator
discriminator = Discriminator().cuda()

# ------------------ Load Pretrained Encoders ------------------

# Load the pretrained encoders (assuming the models were saved after pretraining)
encoder_hq.load_state_dict(torch.load('encoder_hq_resnet34_pretrained.pth'))
encoder_lq.load_state_dict(torch.load('encoder_lq_resnet34_pretrained.pth'))

# ------------------ Optimizers ------------------

optimizer_encoder = optim.Adam(list(encoder_hq.parameters()) + list(encoder_lq.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_generator = optim.Adam(stylegan_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ------------------ Loss Functions ------------------

adversarial_loss = nn.BCELoss()  # Binary Cross-Entropy Loss for the discriminator
reconstruction_loss = nn.L1Loss()  # L1 loss for image reconstruction

# ------------------ Training Loop ------------------

num_epochs = 20
for epoch in range(num_epochs):
    encoder_hq.train()
    encoder_lq.train()
    stylegan_generator.train()
    discriminator.train()

    total_loss = 0
    for hq_images, lq_images in zip(hq_train_loader, lq_train_loader):
        hq_images = hq_images.cuda()
        lq_images = lq_images.cuda()

        # ------------------ Train the Discriminator ------------------
        real_labels = torch.ones(hq_images.size(0), 1).cuda()
        fake_labels = torch.zeros(hq_images.size(0), 1).cuda()

        # Real images (HQ)
        real_loss = adversarial_loss(discriminator(hq_images), real_labels)

        # Generated images (from LQ images)
        w_l = encoder_lq(lq_images)  # Encode LQ images
        reconstructed_hq = stylegan_generator([w_l, torch.zeros(w_l.size(0), 0).cuda()])  # Generate from latent vector
        fake_loss = adversarial_loss(discriminator(reconstructed_hq.detach()), fake_labels)

        # Total Discriminator loss
        d_loss = real_loss + fake_loss
        optimizer_discriminator.zero_grad()
        d_loss.backward()
        optimizer_discriminator.step()

        # ------------------ Train the Generator and Encoders ------------------

        # Adversarial loss for the generator
        g_adv_loss = adversarial_loss(discriminator(reconstructed_hq), real_labels)

        # Reconstruction loss
        g_rec_loss = reconstruction_loss(reconstructed_hq, hq_images)

        # Total Generator loss
        g_loss = g_adv_loss + g_rec_loss
        optimizer_generator.zero_grad()
        optimizer_encoder.zero_grad()
        g_loss.backward()
        optimizer_generator.step()
        optimizer_encoder.step()

        total_loss += g_loss.item()

        # Log the losses in the required format
        D_h2l = real_loss.item()
        D_l2h = fake_loss.item()
        E_h2l = g_adv_loss.item()
        E_l2h = g_rec_loss.item()

        print(f"D_h2l: {D_h2l:.3f}, D_l2h: {D_l2h:.3f}, E_h2l: {E_h2l:.3f}, E_l2h: {E_l2h:.3f}")

    print(f"Epoch {epoch+1}, Generator Loss: {total_loss/len(hq_train_loader)}")

# ------------------ Save the Final Model ------------------

# Save the final model (encoder, generator, and discriminator)
torch.save(encoder_hq.state_dict(), 'encoder_hq_final.pth')
torch.save(encoder_lq.state_dict(), 'encoder_lq_final.pth')
torch.save(stylegan_generator.state_dict(), 'stylegan_generator_final.pth')
torch.save(discriminator.state_dict(), 'discriminator_final.pth')
