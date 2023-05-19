
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary
import random
from zipfile import ZipFile
import requests
import colour
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear,mosaicing_CFA_Bayer,demosaicing_CFA_Bayer_Menon2007
class DMCNN_VD(torch.nn.Module):
    def __init__(self, n_layers=20):
        super(DMCNN_VD, self).__init__()

        self.n_layers = n_layers
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.SELU()
        )
        for i in range(1, self.n_layers):
            setattr(self, f'layer{i}', self.conv_layer)

        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(3),
            torch.nn.SELU(inplace=True)
            #torch.nn.Dropout2d(p = 0.25)
        )
        self.apply(self._msra_init)
            
    def forward(self, x):
        out = getattr(self, 'layer0')(x)
        for i in range(1, self.n_layers):
            out = getattr(self, f'layer{i}')(out)

        out = self.residual(out)

        return out
    
    @property
    def conv_layer(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.SELU()
        )

    @property
    def n_params(self):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _msra_init(self, m):
        if isinstance(m, torch.nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2./n))
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

class ImagePatchDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, loader=None, sample_size=None, 
                 sample_idx=None, patch_size=(33, 33), bilin=True):
        self.root = root
 
        self.transform = transform
        if not self.transform:
            self.transform = torch.from_numpy

        self.patch_size = patch_size
        self.bilin = bilin

        self.loader = loader
        if not self.loader:
            self.loader = self._numpy_loader

        self.sample_size = sample_size
        files = os.listdir(root)

        self.files_ = list(map(lambda x: os.path.join(root, x), files))
        self.images_ = list(map(lambda x: np.array(self.loader(x)), self.files_))
        self.cfa_ = list(map(self._mosaic, self.images_))
        self.patches_ = self._compute_patches(self.images_)
        if self.bilin:
            self.bilinears_ = list(map(self._bilin, self.cfa_))
        
    def __getitem__(self, idx):
        patch, img_id = self.patches_[idx]

        x, y = patch
        b0, b1 = self.patch_size
        truth = self.images_[img_id][x - b0:x, y - b1:y, :]
        cfa = self.cfa_[img_id][x - b0:x, y - b1:y].reshape((3, 33, 33))

        truth = truth.reshape((3, 33, 33))

        if self.bilin:
            bilin = self.bilinears_[img_id][x - b0:x, y- b1:y, :].reshape((3, 33, 33))

        if self.transform:
            truth = self.transform(truth)
            cfa = self.transform(cfa)
            if self.bilin:
                bilin = self.transform(bilin)

        if self.bilin:
            return cfa, truth, bilin

        return cfa, truth

    def __len__(self):
        return len(self.patches_)

    def _compute_patches(self, images):

        patches = []
        for idx, img in enumerate(images):
            image_patch = []
            H, W, C = img.shape
            b0, b1 = self.patch_size

            for i in range(b0, H-b0, 5):
                for j in range(b1, W-b1, 5):
                    image_patch.append(([i, j], idx))
            
            if self.sample_size:
                image_patch = random.sample(image_patch, self.sample_size)
            patches += image_patch

        return patches

    def _numpy_loader(self, path):
        return colour.io.read_image(path)
        #return cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2RGB)

    def _mosaic(self, img):

        noise = np.zeros(img.shape, img.dtype)
        cv2.randn(noise, (15, 15, 15), (5, 5, 5))
        noisy_img = cv2.add(img * 255.0, noise)
        noisy_img = (noisy_img / 255.0).astype(np.float64)
        cfa = np.zeros(img.shape, np.float64)
        mosaic = mosaicing_CFA_Bayer(noisy_img)
        for i in range(3):
            cfa[:, :, i] = mosaic
        
        return cfa

    def _bilin(self, cfa):

        bilin = demosaicing_CFA_Bayer_Menon2007(cfa[:, :, 0])
        #denoised_img = cv2.GaussianBlur(bilin, (9, 9), 0)
        return bilin

def load_checkpoint(path, model, optimizer, loss, **kwargs):

    params = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        loss=loss
    )
    params.update(kwargs)
    
    torch.save(params, path)
def add_gaussian_noise(image, mean=0, std=5):
    """向图像中添加高斯噪声"""
    mean = 0           # 均值
    var = 20           # 方差
    sigma = var ** 0.5 # 标准差
    gaussian = np.random.normal(mean, sigma, image.shape) # 生成正态分布的噪声
    gaussian = gaussian.reshape(image.shape[0],image.shape[1],image.shape[2])
    noisy_image = np.clip(image * 255 + gaussian, 0, 255)
    #noise = np.zeros(image.shape, image.dtype)
    #cv2.randn(noise, (15, 15, 15), (5, 5, 5))
    #image = image * 255
    #noisy_image = cv2.add(image, noise)
    #noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image_float32 = (noisy_image).astype(np.float32)
    #noisy_image_bgr = cv2.cvtColor(noisy_image_float32, cv2.COLOR_RGB2BGR)
    return noisy_image_float32
def calculate_psnr(img1, img2):
    # 检查图像尺寸是否相同
    if img1.shape != img2.shape:
        raise ValueError("输入的图像尺寸不匹配")

    # 计算MSE
    mse = np.mean((img1 - img2) ** 2)

    # 计算最大像素值
    max_pixel = np.max(img1)

    # 计算PSNR
    psnr = 20 * np.log10(1) - 10 * np.log10(mse)

    return psnr
def showprocess(true, mosaic, bilin, demosaiced, id):
    # 显示4个图像
    plt.subplot(2, 2, 1)
    plt.imshow(true)
    plt.title('true')
    #true_float32 = (true * 255).astype(np.float32)
    #true_bgr = cv2.cvtColor(true_float32, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('true/Img{:03d}.png'.format(id), true_bgr)

    plt.subplot(2, 2, 2)
    plt.imshow(mosaic, cmap='gray')
    plt.title('mosaic')
    #mosaic_float32 = (mosaic * 255).astype(np.float32)
    #mosaic_bgr = cv2.cvtColor(mosaic_float32, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('mosaic/Img{:03d}.png'.format(id), mosaic_bgr)

    plt.subplot(2, 2, 3)
    plt.imshow(bilin, cmap='gray')
    plt.title('bilin')
    bilin_float32 = (bilin * 255).astype(np.float32)
    bilin_bgr = cv2.cvtColor(bilin_float32, cv2.COLOR_RGB2BGR)
    cv2.imwrite('bilin/Img{:03d}.png'.format(id), bilin_bgr)

    plt.subplot(2, 2, 4)
    plt.imshow(demosaiced, cmap='gray')
    plt.title('demosaic')
    demosaiced_float32 = (demosaiced * 255).astype(np.float32)
    demosaiced_bgr = cv2.cvtColor(demosaiced_float32, cv2.COLOR_RGB2BGR)
    cv2.imwrite('demosaiced/Img{:03d}.png'.format(id), demosaiced_bgr)

    # 调整子图间距
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    #save_images(true, mosaic, bilin, demosaiced, id)
    # 显示窗口
    plt.show()
    #plt.imshow(true)
    #plt.title('true')
    #plt.show()

    ## 显示mosaic
    #plt.imshow(mosaic, cmap='gray')
    #plt.title('mosaic')
    #plt.show()

    ## 显示bilin
    #plt.imshow(bilin, cmap='gray')
    #plt.title('bilin')
    #plt.show()

    ## 显示demosaic
    #plt.imshow(demosaiced, cmap='gray')
    #plt.title('demosaic')
    #plt.show()
def save_images(true, mosaic, bilin, demosaiced, id):
    # 设置文件名
    img_format = '{:s}/Img{:03d}_{:s}.png'
    # 将图像存储到对应的文件夹中
    Image.fromarray((mosaic*255).astype(np.uint8)).save(img_format.format('mosaic', id, 'mosaic'))
    Image.fromarray((bilin*255).astype(np.uint8)).save(img_format.format('bilin', id, 'bilin'))
    Image.fromarray((demosaiced*255).astype(np.uint8)).save(img_format.format('demosaiced', id, 'demosaiced'))
    print('保存完成')

def testing_model(mymodel, img_path, id):
    myimg = colour.io.read_image(img_path)
    mymodel = mymodel.eval()

    #noise = np.zeros(myimg.shape, myimg.dtype)
    #cv2.randn(noise, (15, 15, 15), (5, 5, 5))
    #noisy_img = cv2.add(myimg * 255.0, noise)
    #noisy_img = (noisy_img / 255.0).astype(np.float64)
    #noisy_img = np.clip(noisy_img, 0.0, 255.0).astype(np.float64)
    noisy_img = add_gaussian_noise(myimg)
    noisy_img = noisy_img.astype(np.float64) / 255.0

    mosaic = mosaicing_CFA_Bayer(noisy_img)

    
    bilin = demosaicing_CFA_Bayer_Menon2007(mosaic)
    #denoised_img = cv2.GaussianBlur(bilin, (9, 9), 0)

    print(calculate_psnr(bilin,myimg))
    denoised_img = bilin


    image_patches = []
    patches = []
    bilin_patches = []
    for i in range(33, mosaic.shape[0], 33):
        for j in range(33, mosaic.shape[1], 33):
            patch_mosaic = mosaic[i-33:i, j-33:j]
            patch = np.zeros((33, 33, 3), np.float64)
            for idx in range(3):
                patch[:, :, idx] = patch_mosaic
            patches.append((i, j))
            image_patches.append(patch)
            bilin_patches.append(denoised_img[i-33:i, j-33:j, :])

    if mosaic.shape[1] % 33 != 0:
        for i in range(33, mosaic.shape[0], 33):
            patch_mosaic = mosaic[i-33:i, mosaic.shape[1]-33:mosaic.shape[1]]
            patch = np.zeros((33, 33, 3), np.float64)
            for idx in range(3):
                patch[:, :, idx] = patch_mosaic
            patches.append((i, mosaic.shape[1]))
            image_patches.append(patch)
            bilin_patches.append(denoised_img[i-33:i, mosaic.shape[1]-33:mosaic.shape[1], :])

    if mosaic.shape[0] % 33 != 0:
        for j in range(33, mosaic.shape[1], 33):
            patch_mosaic = mosaic[mosaic.shape[0]-33:mosaic.shape[0], j-33:j]
            patch = np.zeros((33, 33, 3), np.float64)
            for idx in range(3):
                patch[:, :, idx] = patch_mosaic
            patches.append((mosaic.shape[0], j))
            image_patches.append(patch)
            bilin_patches.append(denoised_img[mosaic.shape[0]-33:mosaic.shape[0], j-33:j, :])
    
    if mosaic.shape[0] % 33 != 0 and mosaic.shape[1] % 33 != 0:
        patch_mosaic = mosaic[mosaic.shape[0]-33:mosaic.shape[0], mosaic.shape[1]-33:mosaic.shape[1]]
        patch = np.zeros((33, 33, 3), np.float64)
        for idx in range(3):
            patch[:, :, idx] = patch_mosaic
        patches.append((mosaic.shape[0], mosaic.shape[1]))
        image_patches.append(patch)
        bilin_patches.append(denoised_img[mosaic.shape[0]-33:mosaic.shape[0], mosaic.shape[1]-33:mosaic.shape[1], :])

    tensor_patches = torch.from_numpy(np.stack(image_patches).reshape((len(image_patches), 3, 33, 33))).float().to(device)
    results = []
    with torch.no_grad():
        for i in range(0, tensor_patches.shape[0]):
            input_patch = tensor_patches[i:i+128, :, :, :]
            outputs = mymodel(input_patch)
            results.append(outputs)
    results = torch.cat(results)

    demosaiced = np.zeros(myimg.shape, np.float64)
    for idx, patch in enumerate(patches):
        i, j = patch
        demosaic_patch = results[idx, :, :, :].reshape(33, 33, 3)
        demosaic_patch = np.array(demosaic_patch.tolist())
        demosaic_patch = np.clip(demosaic_patch, 0, 1.0)  # 将像素点限定在0~1之间
        #demosaic_patch = (demosaic_patch).astype(np.float64)
        demosaiced_patch = np.add(demosaic_patch, bilin_patches[idx].astype(np.float64))  # 使用np.add()函数进行加法运算，并将类型转换为浮点型
        #demosaiced_patch = demosaic_patch
        #demosaic_patch = np.clip(demosaic_patch, 0.0, 1.0).astype(np.float64)  # 再次使用np.clip()函数限定像素点的范围
        demosaiced[i-33:i, j-33:j, :] = demosaiced_patch.astype(np.float64)
    demosaiced = np.clip(demosaiced, 0.0, 1.0).astype(np.float64)  # 将像素点的范围限定在正确值范围内 
    print(calculate_psnr(bilin,myimg))
    print(calculate_psnr(demosaiced,myimg))
    showprocess(myimg, mosaic, denoised_img, demosaiced, id)
    #showprocess((myimg*255).astype(np.uint8), mosaic*255, (denoised_img*255).astype(np.uint8), (demosaiced*255).astype(np.uint8), id)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)



model_vd = DMCNN_VD().cuda()
summary(model_vd, input_size=(3, 33, 33))

criterion = torch.nn.MSELoss()

optimizer_vd = torch.optim.Adam(model_vd.parameters(), lr=1e-5)
#scheduler = ReduceLROnPlateau(optimizer_vd, mode='min', factor=0.1, patience=1)
#加载原有模型
weights_path = 'weights'
if not os.path.exists(weights_path):
    os.mkdir(weights_path)
checkpoint_path = 'dmdncnn_vd_5.weight'
checkpoint_path = os.path.join(weights_path, checkpoint_path) if checkpoint_path else checkpoint_path
print(checkpoint_path)
if checkpoint_path and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_vd.load_state_dict(checkpoint['model_state_dict'])
    optimizer_vd.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']


#开始训练
device = torch.device('cuda:0')
n_epochs = 30
start_epoch = 6
loss_list = []

#if torch.cuda.is_available():
#    print('GPU is available')
#    print('Current device :', torch.cuda.current_device())
#else:
#    print('CPU is used')

flag = input("是否进行训练(Y/N)：")
if flag.lower() == "y":
    # 执行train操作
    dataset = ImagePatchDataset("data", patch_size=(33, 33))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    total_step = len(data_loader)
    
    for epoch in range(start_epoch, n_epochs):
        epoch_loss = []
        for idx, (cfa, target, bilin) in enumerate(data_loader):
            cfa = cfa.float().to(device)
            target = target.float().to(device)
            bilin = bilin.float().to(device)
            target.cuda()
            outputs = model_vd(cfa)
        
            loss = criterion(outputs + bilin, target)
            epoch_loss.append(loss.item())

            optimizer_vd.zero_grad()
            loss.backward()
            optimizer_vd.step()

            if idx % 50 == 0:
                print(f'Epoch [{epoch}/{n_epochs}], Step [{idx}/{total_step}], Loss: {loss.item()}')
        epoch_stats = np.array(epoch_loss)
        print(f'\nFinished Epoch {epoch}, Loss --- mean: {epoch_stats.mean()}, std {epoch_stats.std()}\n')
        psnr = 20 * np.log10(255.0) - 10 * np.log10(epoch_stats.mean() * 255)
        with open("output.txt", "a") as f:
            f.write(f'\nEpoch {epoch}, psnr = {psnr}\n')
        loss_list.append(epoch_stats.mean())
        #scheduler.step(epoch_stats.mean())

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,8))
        #print(np.array(outputs[-1].tolist()).reshape((33, 33, 3)))
        ax1.imshow(np.array(outputs[-1].tolist()).reshape((33, 33, 3)) + np.array(bilin[-1].tolist()).reshape((33, 33, 3)))
        ax1.set_title('CNN')
        ax2.imshow(np.array(cfa[-1].tolist()).reshape((33, 33, 3)), cmap='gray')
        ax2.set_title('CFA')
        ax3.imshow(np.array(target[-1].tolist()).reshape((33, 33, 3)))
        ax3.set_title('Ground truth')
        ax4.imshow(np.array(bilin[-1].tolist()).reshape((33, 33, 3)))
        ax4.set_title('Bilinear')

        load_checkpoint(os.path.join(weights_path, f'dmdncnn_vd_{epoch}.weight'), model_vd, optimizer_vd, loss, epoch=epoch)
        plt.show()



img_dir = 'Kodak24/kodim{:02d}.png'  # 图片文件名的格式
if not os.path.exists('mosaic'):
    os.mkdir('mosaic')
if not os.path.exists('bilin'):
    os.mkdir('bilin')
if not os.path.exists('demosaiced'):
    os.mkdir('demosaiced')

while True:
    print('请输入你要处理的图片编号：')
    #i = int(input())
    
    for i in range(1, 19):
        print('当前处理的图片编号为：', i)
        img_path = img_dir.format(i)
        testing_model(model_vd, img_path, i)
    
    ans = input("是否继续进行处理？(Y/N)")  # 询问用户是否继续处理
    if ans in ('N', 'n'):
        print('程序已退出')
        break



