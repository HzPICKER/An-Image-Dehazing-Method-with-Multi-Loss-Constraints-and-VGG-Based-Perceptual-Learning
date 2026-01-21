import tensorflow as tf
import keras
import keras.backend as K
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import re
import time
from keras.preprocessing.image import img_to_array
from tensorflow.keras import layers, Model
import psutil
import os

# Define the size of the image
SIZE = 256
clean_img = []
hazy_img_dict = {}

example_clean = []
example_hazy = []
clean_filenames = []
hazy_filenames = []

# Load clean images
path_clear = 'dehaze/clear_images'
files_clear = sorted(os.listdir(path_clear))  # Sort to ensure order
for i in tqdm(files_clear):
    img = cv2.imread(os.path.join(path_clear, i), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.astype('float32') / 255.0
    clean_img.append(img)
    clean_filenames.append(i)  # Store the filename

# Load hazy images and organize them by their corresponding clear image
path_hazy = 'dehaze/haze'
files_hazy = sorted(os.listdir(path_hazy))  # Sort to ensure order
for i in tqdm(files_hazy):
    base_name = os.path.splitext(i)[0].split('_')[0]
    if base_name not in hazy_img_dict:
        hazy_img_dict[base_name] = []
    if len(hazy_img_dict[base_name]) < 3:  # Limit to 3 hazy images per clean image
        img = cv2.imread(os.path.join(path_hazy, i), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        hazy_img_dict[base_name].append(img_to_array(img))
        hazy_filenames.append(i)  # Store the filename

# Flatten the hazy image list and match with clean images
matching_clean_img = []
matching_hazy_img = []
epoch_times = []

for i, img in enumerate(clean_img):
    base_name = os.path.splitext(files_clear[i])[0]  # Extract base name (like '0468' from '0468.jpg')
    if base_name in hazy_img_dict:
        for hazy_img in hazy_img_dict[base_name]:
            matching_clean_img.append(img)  # Repeat the clean image for each corresponding hazy image
            matching_hazy_img.append(hazy_img)

# Check if the number of clean images and hazy images match
print(f"Number of clean images: {len(matching_clean_img)}")
print(f"Number of hazy images: {len(matching_hazy_img)}")

assert len(matching_clean_img) == len(
    matching_hazy_img), "Mismatch in the number of clean and hazy images after filtering."

# Create TensorFlow datasets
clean_dataset = tf.data.Dataset.from_tensor_slices(np.array(matching_clean_img))
hazy_dataset = tf.data.Dataset.from_tensor_slices(np.array(matching_hazy_img))

# Combine hazy and clean datasets
dataset = tf.data.Dataset.zip((hazy_dataset, clean_dataset))

# THIS IS FOR TESTING DATA
# defining the size of the image
SIZE = 256
test_clean_img = []
path = 'filtered-dehaze/filtered/test/clear'
files = os.listdir(path)
for i in tqdm(files):
    img = cv2.imread(path + '/' + i, 1)
    # open cv reads images in BGR format so we have to convert it to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resizing image
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.astype('float32') / 255.0
    test_clean_img.append(img_to_array(img))

test_hazy_img = []
path = 'filtered-dehaze/filtered/test/hazy'
files = os.listdir(path)
for i in tqdm(files):
    img = cv2.imread(path + '/' + i, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resizing image
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.astype('float32') / 255.0
    test_hazy_img.append(img_to_array(img))

# Batch the datasets
clean_dataset = clean_dataset.batch(1)
hazy_dataset = hazy_dataset.batch(1)

test_data_clean = tf.data.Dataset.from_tensor_slices(np.array(test_clean_img)).batch(16)
test_data_hazy = tf.data.Dataset.from_tensor_slices(np.array(test_hazy_img)).batch(16)

example_clean = next(iter(clean_dataset))
example_hazy = next(iter(hazy_dataset))

for example_input, example_target in tf.data.Dataset.zip((hazy_dataset, clean_dataset)).take(1):
    print("Shape of example_input:", example_input.shape)
    print("Shape of example_target:", example_target.shape)


# ============================================
# 数据增强功能（用于提升模型性能和减少过拟合）
# ============================================

def data_augmentation(hazy, clean):
    """
    对输入图像进行数据增强
    有助于提升模型泛化能力，减少过拟合
    """
    # 随机水平翻转（50%概率）
    if tf.random.uniform([]) > 0.5:
        hazy = tf.image.flip_left_right(hazy)
        clean = tf.image.flip_left_right(clean)

    # 随机亮度调整（仅对hazy图像，模拟不同光照条件）
    hazy = tf.image.random_brightness(hazy, max_delta=0.1)

    # 随机对比度调整（仅对hazy图像）
    hazy = tf.image.random_contrast(hazy, lower=0.9, upper=1.1)

    # 确保像素值在[0,1]范围内
    hazy = tf.clip_by_value(hazy, 0.0, 1.0)
    clean = tf.clip_by_value(clean, 0.0, 1.0)

    return hazy, clean


# ============================================
# VGG感知损失类
# ============================================

class VGGPerceptualLoss(tf.keras.losses.Loss):
    def __init__(self, weight=0.1, vgg_layers=['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3'],
                 **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

        # 加载预训练的VGG19模型（不含顶层）
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        # 获取指定层的输出
        outputs = [vgg.get_layer(layer).output for layer in vgg_layers]
        self.model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

        # 冻结VGG模型的权重
        self.model.trainable = False

        # VGG预处理函数
        self.preprocess = tf.keras.applications.vgg19.preprocess_input

    def call(self, y_true, y_pred):
        # 确保输入在[0,1]范围内
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

        # 将输入从[0,1]转换到[0,255]，然后进行VGG预处理
        y_true_processed = self.preprocess(y_true * 255.0)
        y_pred_processed = self.preprocess(y_pred * 255.0)

        # 提取特征
        true_features = self.model(y_true_processed)
        pred_features = self.model(y_pred_processed)

        # 计算感知损失（特征图之间的L1距离）
        loss = 0
        for f_true, f_pred in zip(true_features, pred_features):
            loss += tf.reduce_mean(tf.abs(f_true - f_pred))

        return loss * self.weight / len(true_features)


# ============================================
# 颜色一致性损失
# ============================================

def color_constancy_loss(images, weight=0.05):
    """保持图像颜色一致性，防止过饱和"""
    # 计算RGB通道的均值
    mean_r = tf.reduce_mean(images[..., 0])
    mean_g = tf.reduce_mean(images[..., 1])
    mean_b = tf.reduce_mean(images[..., 2])

    # 惩罚RGB均值之间的差异
    rg_diff = tf.square(mean_r - mean_g)
    gb_diff = tf.square(mean_g - mean_b)
    br_diff = tf.square(mean_b - mean_r)

    return weight * (rg_diff + gb_diff + br_diff)


# ============================================
# 改进的损失函数
# ============================================

def adaptive_loss_weights(epoch, total_epochs):
    """动态调整各损失项的权重"""
    progress = epoch / total_epochs

    if progress < 0.3:  # 初期：强去雾
        lambda_l1 = 80
        lambda_ssim = 0.1
        lambda_perceptual = 0.05
        lambda_color = 0.05
    elif progress < 0.7:  # 中期：平衡
        lambda_l1 = 60
        lambda_ssim = 0.15
        lambda_perceptual = 0.1
        lambda_color = 0.1
    else:  # 后期：自然度
        lambda_l1 = 40
        lambda_ssim = 0.2
        lambda_perceptual = 0.15
        lambda_color = 0.15

    return lambda_l1, lambda_ssim, lambda_perceptual, lambda_color


def upsample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                               padding='same',
                                               kernel_initializer='he_normal',
                                               use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result


class ReshapeLayer(layers.Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, self.target_shape)

    def get_config(self):
        config = super(ReshapeLayer, self).get_config()
        config.update({'target_shape': self.target_shape})
        return config


class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = tf.random_normal_initializer(mean=0., stddev=1.)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsules, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')
        self.dense = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsules, 1, 1])
        inputs_tiled = tf.expand_dims(inputs_tiled, 4)

        inputs_hat = tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled)

        b_ij = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsules, self.input_num_capsule, 1, 1])

        for i in range(self.routings):
            c_ij = tf.nn.softmax(b_ij, axis=1)
            s_j = tf.multiply(c_ij, inputs_hat)
            s_j = tf.reduce_sum(s_j, axis=2, keepdims=True)
            v_j = squash(s_j)
            if i < self.routings - 1:
                b_ij += tf.matmul(inputs_hat, v_j, transpose_a=True)

        v_j = tf.squeeze(v_j, axis=2)
        output = self.dense(v_j)
        return output

    def get_config(self):
        config = {
            'num_capsules': self.num_capsules,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings,
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def squash(x, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(x), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return scale * x


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    # Multi-scale downsampling
    down_stack_1 = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    down_stack_2 = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
    ]

    # Multi-scale upsampling
    up_stack_1 = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    up_stack_2 = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(256, 4),  # (bs, 8, 8, 512)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    # Multi-scale processing
    x1 = inputs
    x2 = inputs

    # Downsampling through the model
    skips_1 = []
    for down in down_stack_1:
        x1 = down(x1)
        skips_1.append(x1)

    skips_2 = []
    for down in down_stack_2:
        x2 = down(x2)
        skips_2.append(x2)

    skips_1 = reversed(skips_1[:-1])
    skips_2 = reversed(skips_2[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack_1, skips_1):
        x1 = up(x1)
        x1 = tf.keras.layers.Concatenate()([x1, skip])

    for up, skip in zip(up_stack_2, skips_2):
        x2 = up(x2)
        x2 = tf.keras.layers.Concatenate()([x2, skip])

    x = tf.keras.layers.Concatenate()([x1, x2])
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    # Convert conv features to capsule
    pre_capsule = ReshapeLayer((-1, down3.shape[1] * down3.shape[2], 256))(
        down3)  # Reshape to [batch_size, num_capsules, dim_capsule]
    capsules = CapsuleLayer(num_capsules=10, dim_capsule=8, routings=3)(pre_capsule)
    capsules_flattened = layers.Flatten()(capsules)

    # Following layers can be adjusted as needed. Here, just an example to proceed from capsules:
    dense1 = layers.Dense(512, activation='relu')(capsules_flattened)
    dense2 = layers.Dense(1, activation='sigmoid')(dense1)

    return tf.keras.Model(inputs=[inp, tar], outputs=dense2)


generator = Generator()
generator.summary()

discriminator = Discriminator()
discriminator.summary()

# 初始化VGG感知损失
vgg_perceptual_loss = VGGPerceptualLoss(weight=0.1)

# loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_object = tf.keras.losses.BinaryCrossentropy()

# 修改原来的优化器定义
initial_learning_rate_gen = 3e-4
initial_learning_rate_disc = 1e-4

# 使用学习率调度
lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate_gen,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate_disc,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

generator_optimizer = tf.keras.optimizers.Adam(lr_schedule_gen, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr_schedule_disc, beta_1=0.5)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def train_step(input_image, target, epoch, total_epochs=20):
    """增强的训练步骤，包含多个损失项"""

    # 获取动态权重
    lambda_l1, lambda_ssim, lambda_perceptual, lambda_color = adaptive_loss_weights(epoch, total_epochs)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        # 判别器输出
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # ========== 生成器损失项 ==========
        # 1. GAN损失
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # 2. L1损失
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        # 3. SSIM损失（基于区域）
        def compute_ssim_loss(img1, img2, patch_size=64):
            patches1 = tf.image.extract_patches(
                img1, sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size // 2, patch_size // 2, 1],
                rates=[1, 1, 1, 1], padding='SAME'
            )
            patches2 = tf.image.extract_patches(
                img2, sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size // 2, patch_size // 2, 1],
                rates=[1, 1, 1, 1], padding='SAME'
            )

            batch_size = tf.shape(img1)[0]
            patch_count = tf.shape(patches1)[1] * tf.shape(patches1)[2]

            patches1 = tf.reshape(patches1, [batch_size * patch_count, patch_size, patch_size, 3])
            patches2 = tf.reshape(patches2, [batch_size * patch_count, patch_size, patch_size, 3])

            ssim_values = tf.image.ssim(patches1, patches2, max_val=1.0)
            return 1.0 - tf.reduce_mean(ssim_values)

        ssim_loss = compute_ssim_loss(target, gen_output, patch_size=64)

        # 4. 感知损失
        perc_loss = vgg_perceptual_loss(target, gen_output)

        # 5. 颜色一致性损失
        color_loss = color_constancy_loss(gen_output, weight=lambda_color)

        # 总生成器损失
        gen_total_loss = (
                gan_loss +
                lambda_l1 * l1_loss +
                lambda_ssim * ssim_loss +
                perc_loss +
                color_loss
        )

        # 判别器损失
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # 梯度裁剪
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    generator_gradients = [tf.clip_by_norm(g, 5.0) for g in generator_gradients]  # 增大裁剪阈值

    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_gradients = [tf.clip_by_norm(g, 5.0) for g in discriminator_gradients]

    # 应用梯度
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    # 计算评价指标
    psnr_value = tf.image.psnr(target, gen_output, max_val=1.0)
    ssim_value = tf.image.ssim(target, gen_output, max_val=1.0)

    return {
        'total_loss': gen_total_loss,
        'gan_loss': gan_loss,
        'l1_loss': l1_loss,
        'ssim_loss': ssim_loss,
        'perceptual_loss': perc_loss,
        'color_loss': color_loss,
        'disc_loss': disc_loss,
        'psnr': psnr_value,
        'ssim': ssim_value
    }


def save_checkpoint(generator, discriminator, checkpoint_dir):
    generator.save_weights(checkpoint_dir + '/generator_weights.weights.h5')
    discriminator.save_weights(checkpoint_dir + '/discriminator_weights.weights.h5')


# Define a function to load a model checkpoint
def load_checkpoint(generator, discriminator, checkpoint_dir):
    generator.load_weights(checkpoint_dir + '/generator_weights.weights.h5')
    discriminator.load_weights(checkpoint_dir + '/discriminator_weights.weights.h5')


# Define a function to save the best generator based on the lowest generator loss
def save_best_generator(generator, best_generator_dir):
    generator.save(best_generator_dir)


def plot_training_curves(gen_loss, disc_loss, val_loss,
                         train_psnr, train_ssim, val_psnr, val_ssim,
                         additional_losses=None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss curves
    axes[0, 0].plot(gen_loss, label='Generator Loss')
    axes[0, 0].plot(disc_loss, label='Discriminator Loss')
    axes[0, 0].plot(val_loss, label='Validation Loss', linestyle='--')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # PSNR curves
    axes[0, 1].plot(train_psnr, label='Train PSNR')
    axes[0, 1].plot(val_psnr, label='Validation PSNR', linestyle='--')
    axes[0, 1].set_title('PSNR Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # SSIM curves
    axes[0, 2].plot(train_ssim, label='Train SSIM')
    axes[0, 2].plot(val_ssim, label='Validation SSIM', linestyle='--')
    axes[0, 2].set_title('SSIM Curves')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('SSIM')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # 如果有额外的损失项，绘制它们
    if additional_losses:
        axes[1, 0].plot(additional_losses['l1_loss'], label='L1 Loss', linestyle='-.')
        axes[1, 0].plot(additional_losses['perceptual_loss'], label='Perceptual Loss', linestyle='-.')
        axes[1, 0].plot(additional_losses['color_loss'], label='Color Loss', linestyle='-.')
        axes[1, 0].set_title('Additional Losses')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        axes[1, 0].axis('off')

    # Gap between train and validation PSNR
    epochs = range(len(gen_loss))
    axes[1, 1].plot(epochs, np.array(train_psnr) - np.array(val_psnr),
                    label='PSNR Gap', color='red')
    axes[1, 1].set_title('Train vs Validation PSNR Gap')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('PSNR Gap (dB)')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].grid(True)

    # Empty subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


def generate_and_save_images(model, epoch, test_input, target, save_dir="training_progress"):
    """生成并保存训练过程中的示例图像"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    prediction = model(test_input, training=False)

    plt.figure(figsize=(12, 4))

    display_list = [
        (test_input[0] + 1) / 2,  # 输入图像
        (target[0] + 1) / 2,  # 目标图像
        (prediction[0] + 1) / 2  # 预测图像
    ]

    titles = ['Hazy Input', 'Clear Target', 'Generated Output']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(np.clip(display_list[i], 0, 1))
        plt.axis('off')

    # 计算当前图像的指标
    psnr_val = tf.image.psnr(target[0:1], prediction[0:1], max_val=1.0).numpy()[0]
    ssim_val = tf.image.ssim(target[0:1], prediction[0:1], max_val=1.0).numpy()[0]

    plt.suptitle(f'Epoch {epoch + 1} - PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}')
    plt.savefig(f"{save_dir}/epoch_{epoch + 1:03d}.png", dpi=150, bbox_inches='tight')
    plt.close()


def save_checkpoint(generator, discriminator, checkpoint_dir):
    """保存模型检查点"""
    import os

    # 创建目录（如果不存在）
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存权重文件
    generator_weights_path = os.path.join(checkpoint_dir, 'generator_weights.weights.h5')
    discriminator_weights_path = os.path.join(checkpoint_dir, 'discriminator_weights.weights.h5')

    generator.save_weights(generator_weights_path)
    discriminator.save_weights(discriminator_weights_path)

    print(f"检查点已保存到: {checkpoint_dir}")


def save_best_generator(generator, best_generator_dir):
    """保存最佳生成器模型"""
    import os

    # 确保目录存在
    dir_path = os.path.dirname(best_generator_dir)
    if dir_path:  # 如果路径包含目录
        os.makedirs(dir_path, exist_ok=True)

    # 使用Keras格式保存，避免HDF5警告
    if best_generator_dir.endswith('.h5'):
        new_path = best_generator_dir.replace('.h5', '.keras')
    else:
        new_path = best_generator_dir + '.keras'

    generator.save(new_path)
    print(f"最佳模型已保存到: {new_path}")


def fit(train_ds, val_ds, epochs, checkpoint_dir='checkpoints', best_generator_dir='best_generator.h5'):
    """修改后的fit函数，支持验证和早停，包含新损失项"""

    import os

    # 创建保存目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 早停参数
    patience = 10
    wait = 0
    best_val_loss = float('inf')

    # 手动学习率调度参数
    lr_patience = 5
    lr_wait = 0
    lr_factor = 0.5

    # 需要将优化器设为全局变量
    global generator_optimizer, discriminator_optimizer

    gen_loss_history = []
    disc_loss_history = []
    val_loss_history = []
    psnr_history = []
    ssim_history = []
    val_psnr_history = []
    val_ssim_history = []

    # 新增：记录各个损失项的历史
    l1_loss_history = []
    perceptual_loss_history = []
    color_loss_history = []

    for epoch in range(epochs):
        start = time.time()
        epoch_gen_loss = []
        epoch_disc_loss = []
        epoch_psnr = []
        epoch_ssim = []

        # 新增：记录各个损失项
        epoch_l1_loss = []
        epoch_perceptual_loss = []
        epoch_color_loss = []

        print(f"Epoch {epoch + 1}/{epochs} 开始...")

        # ========== 训练阶段 ==========
        for input_image, target in train_ds:
            losses = train_step(input_image, target, epoch, epochs)

            epoch_gen_loss.append(losses['total_loss'])
            epoch_disc_loss.append(losses['disc_loss'])
            epoch_psnr.append(tf.reduce_mean(losses['psnr']))
            epoch_ssim.append(tf.reduce_mean(losses['ssim']))

            # 记录各个损失项
            epoch_l1_loss.append(losses['l1_loss'])
            epoch_perceptual_loss.append(losses['perceptual_loss'])
            epoch_color_loss.append(losses['color_loss'])

        # 计算训练集平均损失和指标
        epoch_gen_avg_loss = tf.reduce_mean(epoch_gen_loss)
        epoch_disc_avg_loss = tf.reduce_mean(epoch_disc_loss)
        epoch_psnr_avg = tf.reduce_mean(epoch_psnr)
        epoch_ssim_avg = tf.reduce_mean(epoch_ssim)

        # 计算各个损失项的平均值
        epoch_l1_avg = tf.reduce_mean(epoch_l1_loss)
        epoch_perceptual_avg = tf.reduce_mean(epoch_perceptual_loss)
        epoch_color_avg = tf.reduce_mean(epoch_color_loss)

        # ========== 验证阶段 ==========
        print("在验证集上评估...")
        val_gen_losses = []
        val_psnr_values = []
        val_ssim_values = []

        for val_input, val_target in val_ds:
            val_gen_output = generator(val_input, training=False)

            # 计算验证集的各个损失项
            current_lambda_l1, current_lambda_ssim, current_lambda_perceptual, current_lambda_color = adaptive_loss_weights(
                epoch, epochs)

            # GAN损失
            val_disc_output = discriminator([val_input, val_gen_output], training=False)
            val_gan_loss = loss_object(tf.ones_like(val_disc_output), val_disc_output)

            # L1损失
            val_l1_loss = tf.reduce_mean(tf.abs(val_target - val_gen_output))

            # SSIM损失
            val_ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(val_target, val_gen_output, max_val=1.0))

            # 感知损失
            val_perceptual_loss = vgg_perceptual_loss(val_target, val_gen_output)

            # 颜色一致性损失
            val_color_loss = color_constancy_loss(val_gen_output, weight=current_lambda_color)

            # 总验证损失
            val_gen_loss = (val_gan_loss +
                            current_lambda_l1 * val_l1_loss +
                            current_lambda_ssim * val_ssim_loss +
                            val_perceptual_loss +
                            val_color_loss)

            val_gen_losses.append(val_gen_loss)

            val_psnr = tf.image.psnr(val_target, val_gen_output, max_val=1.0)
            val_ssim = tf.image.ssim(val_target, val_gen_output, max_val=1.0)
            val_psnr_values.append(tf.reduce_mean(val_psnr))
            val_ssim_values.append(tf.reduce_mean(val_ssim))

        val_gen_avg_loss = tf.reduce_mean(val_gen_losses)
        val_psnr_avg = tf.reduce_mean(val_psnr_values)
        val_ssim_avg = tf.reduce_mean(val_ssim_values)

        # ========== 早停机制 ==========
        if val_gen_avg_loss < best_val_loss:
            best_val_loss = val_gen_avg_loss
            wait = 0
            # 保存最佳模型
            print(f"验证损失改善，保存最佳模型到 {best_generator_dir}")
            save_best_generator(generator, best_generator_dir)
        else:
            wait += 1
            if wait >= patience:
                print(f"早停触发，训练停止在epoch {epoch + 1}")
                break

        # ========== 手动学习率调度 ==========
        if val_gen_avg_loss >= best_val_loss:
            lr_wait += 1
            if lr_wait >= lr_patience:
                # 获取当前学习率
                current_gen_lr = float(generator_optimizer.learning_rate.numpy())
                current_disc_lr = float(discriminator_optimizer.learning_rate.numpy())

                # 计算新学习率
                new_gen_lr = max(current_gen_lr * lr_factor, 1e-6)
                new_disc_lr = max(current_disc_lr * lr_factor, 1e-6)

                # 重新创建优化器
                generator_optimizer = tf.keras.optimizers.Adam(new_gen_lr, beta_1=0.5)
                discriminator_optimizer = tf.keras.optimizers.Adam(new_disc_lr, beta_1=0.5)

                print(f"学习率降低: Generator {current_gen_lr:.2e} -> {new_gen_lr:.2e}, "
                      f"Discriminator {current_disc_lr:.2e} -> {new_disc_lr:.2e}")

                lr_wait = 0
        else:
            lr_wait = 0

        # ========== 保存检查点 ==========
        if (epoch + 1) % 5 == 0:  # 每5个epoch保存一次检查点
            epoch_checkpoint_dir = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}')
            save_checkpoint(generator, discriminator, epoch_checkpoint_dir)

        # ========== 记录历史 ==========
        gen_loss_history.append(epoch_gen_avg_loss.numpy())
        disc_loss_history.append(epoch_disc_avg_loss.numpy())
        val_loss_history.append(val_gen_avg_loss.numpy())
        psnr_history.append(epoch_psnr_avg.numpy())
        ssim_history.append(epoch_ssim_avg.numpy())
        val_psnr_history.append(val_psnr_avg.numpy())
        val_ssim_history.append(val_ssim_avg.numpy())

        # 记录各个损失项的历史
        l1_loss_history.append(epoch_l1_avg.numpy())
        perceptual_loss_history.append(epoch_perceptual_avg.numpy())
        color_loss_history.append(epoch_color_avg.numpy())

        # ========== 打印信息 ==========
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        current_gen_lr = float(generator_optimizer.learning_rate.numpy())
        current_disc_lr = float(discriminator_optimizer.learning_rate.numpy())

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  训练损失 - Gen: {epoch_gen_avg_loss:.4f}, Disc: {epoch_disc_avg_loss:.4f}")
        print(f"  验证损失 - Gen: {val_gen_avg_loss:.4f}")
        print(f"  训练指标 - PSNR: {epoch_psnr_avg:.2f} dB, SSIM: {epoch_ssim_avg:.4f}")
        print(f"  验证指标 - PSNR: {val_psnr_avg:.2f} dB, SSIM: {val_ssim_avg:.4f}")
        print(
            f"  损失项详情 - L1: {epoch_l1_avg:.4f}, Perceptual: {epoch_perceptual_avg:.4f}, Color: {epoch_color_avg:.4f}")
        print(f"  学习率 - Gen: {current_gen_lr:.2e}, Disc: {current_disc_lr:.2e}")
        print(f"  时间: {epoch_time:.2f} 秒")
        print(f"  早停等待: {wait}/{patience}, 学习率等待: {lr_wait}/{lr_patience}")
        print("-" * 50)

    # ========== 训练结束，绘制图表 ==========
    additional_losses = {
        'l1_loss': l1_loss_history,
        'perceptual_loss': perceptual_loss_history,
        'color_loss': color_loss_history
    }

    plot_training_curves(
        gen_loss_history, disc_loss_history, val_loss_history,
        psnr_history, ssim_history, val_psnr_history, val_ssim_history,
        additional_losses
    )

    return generator


def check_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"当前内存使用: {memory_usage:.2f} MB")

    system_memory = psutil.virtual_memory()
    print(f"系统内存: {system_memory.percent}% 已使用, {system_memory.available / 1024 / 1024:.2f} MB 可用")


# 使用更小的配置进行测试
def create_train_dataset(batch_size=2, use_augmentation=True):  # 更小的批量大小
    print(f"使用小批量大小: {batch_size}")

    # 使用部分数据进行测试
    test_size = min(100, len(matching_clean_img))  # 最多使用100张图像测试
    clean_subset = matching_clean_img[:test_size]
    hazy_subset = matching_hazy_img[:test_size]

    print(f"使用 {len(clean_subset)} 张图像进行训练测试")

    clean_ds = tf.data.Dataset.from_tensor_slices(np.array(clean_subset))
    hazy_ds = tf.data.Dataset.from_tensor_slices(np.array(hazy_subset))

    train_dataset = tf.data.Dataset.zip((hazy_ds, clean_ds))

    # 应用数据增强（如果启用）
    if use_augmentation:
        train_dataset = train_dataset.map(
            data_augmentation,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        print("数据增强已启用")

    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(100)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset


check_memory_usage()

# 使用小数据集测试
train_dataset = create_train_dataset(batch_size=2)


# 创建验证集（从训练数据中划分）
def split_train_val(dataset, val_split=0.2):
    """将数据集分割为训练集和验证集"""
    total_samples = len(matching_clean_img)
    val_size = int(total_samples * val_split)

    # 随机打乱
    indices = np.random.permutation(total_samples)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # 创建训练集
    train_clean = np.array(matching_clean_img)[train_indices]
    train_hazy = np.array(matching_hazy_img)[train_indices]

    # 创建验证集
    val_clean = np.array(matching_clean_img)[val_indices]
    val_hazy = np.array(matching_hazy_img)[val_indices]

    # 转换为TensorFlow数据集
    train_ds = tf.data.Dataset.from_tensor_slices((train_hazy, train_clean))
    train_ds = train_ds.batch(2).shuffle(100).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_hazy, val_clean))
    val_ds = val_ds.batch(2).prefetch(tf.data.AUTOTUNE)

    print(f"训练集大小: {len(train_clean)}")
    print(f"验证集大小: {len(val_clean)}")

    return train_ds, val_ds


# 分割数据集
train_dataset, val_dataset = split_train_val(None, val_split=0.1)

# 先训练少量epoch测试
print("开始5个epoch的测试训练...")
best_generator = fit(train_ds=train_dataset, val_ds=val_dataset, epochs=5)


def plot_images(a=4, clean_filenames=None, hazy_filenames=None):
    for i in range(a):
        plt.figure(figsize=(10, 10))

        plt.subplot(121)
        plt.title(f'Clean: {clean_filenames[i]}')  # Display the clean image filename
        plt.imshow(example_clean[i])

        plt.subplot(122)
        plt.title(f'Hazy: {hazy_filenames[i]}')  # Display the hazy image filename
        plt.imshow(example_hazy[i])

        plt.show()


# Now call the function and pass the filenames
plot_images(1, clean_filenames=clean_filenames, hazy_filenames=hazy_filenames)


def generate_images(model, test_input, tar):
    """修复后的图像生成函数"""
    prediction = model(test_input, training=False)
    plt.figure(figsize=(15, 5))

    # 确保图像值在[0,1]范围内
    display_list = [
        test_input[0].numpy(),
        tar[0].numpy(),
        np.clip(prediction[0].numpy(), 0, 1)  # 裁剪到[0,1]范围
    ]

    title = ['Hazy Input', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')

    # 计算指标
    psnr = tf.image.psnr(tar[0:1], prediction[0:1], max_val=1.0).numpy()[0]
    ssim = tf.image.ssim(tar[0:1], prediction[0:1], max_val=1.0).numpy()[0]

    plt.suptitle(f'PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}')
    plt.show()

    return psnr, ssim


def calculate_metrics(model, hazy_dataset, clean_dataset, verbose=True):
    """计算模型的PSNR和SSIM指标（更健壮的版本）"""
    all_psnr = []
    all_ssim = []
    total_samples = 0

    # 将两个数据集zip起来
    dataset = tf.data.Dataset.zip((hazy_dataset, clean_dataset))

    for batch_idx, (hazy_batch, clean_batch) in enumerate(dataset):
        batch_size = hazy_batch.shape[0]

        # 生成预测
        prediction = model(hazy_batch, training=False)

        # 计算PSNR和SSIM
        psnr = tf.image.psnr(clean_batch, prediction, max_val=1.0)
        ssim = tf.image.ssim(clean_batch, prediction, max_val=1.0)

        # 将结果转换为numpy
        psnr_np = psnr.numpy()
        ssim_np = ssim.numpy()

        # 添加到列表
        all_psnr.extend(psnr_np.flatten().tolist())
        all_ssim.extend(ssim_np.flatten().tolist())
        total_samples += batch_size

        if verbose:
            print(f"批次 {batch_idx + 1}: {batch_size} 张图像, "
                  f"PSNR范围: [{psnr_np.min():.2f}, {psnr_np.max():.2f}], "
                  f"SSIM范围: [{ssim_np.min():.4f}, {ssim_np.max():.4f}]")

    # 计算平均值
    if all_psnr and all_ssim:
        average_psnr = np.mean(all_psnr)
        average_ssim = np.mean(all_ssim)
        psnr_std = np.std(all_psnr)
        ssim_std = np.std(all_ssim)
    else:
        average_psnr = 0.0
        average_ssim = 0.0
        psnr_std = 0.0
        ssim_std = 0.0

    if verbose:
        print(f"\n总计: {total_samples} 张图像")
        print(f"平均PSNR: {average_psnr:.2f} ± {psnr_std:.2f} dB")
        print(f"平均SSIM: {average_ssim:.4f} ± {ssim_std:.4f}")

    return average_psnr, average_ssim


for example_input, example_target in tf.data.Dataset.zip((hazy_dataset, clean_dataset)).take(2):
    generate_images(generator, example_input, example_target)

# 由于我们已经完成了5个epoch的训练，现在可以：
# 1. 继续训练更多epoch
# 2. 或者在测试集上评估模型

# 建议：先评估模型，然后决定是否继续训练
print("\n" + "=" * 70)
print("5个epoch训练完成！现在评估模型性能...")
print("=" * 70)

# 评估模型
avg_psnr, avg_ssim = calculate_metrics(generator, test_data_hazy, test_data_clean)
print(f"\n测试集评估结果:")
print(f"  PSNR: {avg_psnr:.2f} dB")
print(f"  SSIM: {avg_ssim:.4f}")

# 生成示例图像
print("\n生成示例去雾图像...")
for example_input, example_target in tf.data.Dataset.zip((test_data_hazy, test_data_clean)).take(2):
    generate_images(generator, example_input, example_target)

print("\n继续训练更多epoch...")
# 创建完整的训练集（使用数据增强）
full_train_ds = tf.data.Dataset.from_tensor_slices((matching_hazy_img, matching_clean_img))
full_train_ds = full_train_ds.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
full_train_ds = full_train_ds.batch(4).shuffle(1000).prefetch(tf.data.AUTOTUNE)

# 继续训练更多epoch
print("开始完整训练（20个epoch）...")
best_generator = fit(train_ds=full_train_ds, val_ds=val_dataset, epochs=20)

best_generator.save('improved_generator_with_vgg.h5')

# Calculate the average epoch time
if epoch_times:
    average_epoch_time = sum(epoch_times) / len(epoch_times)
    print(f"Average time per epoch: {average_epoch_time:.2f} seconds")

for example_input, example_target in tf.data.Dataset.zip((test_data_hazy, test_data_clean)).take(2):
    generate_images(generator, example_input, example_target)


def evaluate_on_validation_set(generator, val_hazy_ds, val_clean_ds, num_batches=None):
    """在验证集上评估模型性能"""
    psnr_values = []
    ssim_values = []

    dataset = tf.data.Dataset.zip((val_hazy_ds, val_clean_ds))

    for i, (hazy_batch, clean_batch) in enumerate(dataset):
        if num_batches and i >= num_batches:
            break

        # 生成去雾图像
        generated_batch = generator(hazy_batch, training=False)

        # 计算指标
        batch_psnr = tf.image.psnr(clean_batch, generated_batch, max_val=1.0)
        batch_ssim = tf.image.ssim(clean_batch, generated_batch, max_val=1.0)

        psnr_values.extend(batch_psnr.numpy())
        ssim_values.extend(batch_ssim.numpy())

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"Validation Results - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


# Assuming you have rain_dataset and clean_dataset
average_psnr, average_ssim = calculate_metrics(generator, test_data_hazy, test_data_clean)
print("改进后的模型 - Average PSNR:", average_psnr)
print("改进后的模型 - Average SSIM:", average_ssim)