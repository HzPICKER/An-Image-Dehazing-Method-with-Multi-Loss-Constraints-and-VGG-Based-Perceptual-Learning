#!/usr/bin/env python3
"""
图像去雾模型使用脚本
用法: python dehaze_app.py [输入图像路径] [输出图像路径(可选)]
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import argparse


class DehazeApp:
    """图像去雾应用程序"""

    def __init__(self, model_path='best_generator.keras'):
        """
        初始化去雾模型

        参数:
            model_path: 模型文件路径，默认为 'best_generator.keras'
        """
        print(f"正在加载去雾模型: {model_path}")

        try:
            # 加载模型
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print("✓ 模型加载成功!")
            print(f"  输入形状: {self.model.input_shape}")
            print(f"  输出形状: {self.model.output_shape}")

            # 设置模型输入尺寸
            self.input_size = (256, 256)

        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            print("请确保模型文件存在且格式正确")
            sys.exit(1)

    def preprocess_image(self, image):
        """
        预处理图像，准备输入模型

        参数:
            image: 输入图像 (BGR格式)

        返回:
            processed: 处理后的图像批次
            original_shape: 原始图像尺寸
        """
        # 将BGR转换为RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # 保存原始尺寸
        original_shape = image_rgb.shape[:2]  # (高度, 宽度)

        # 调整到模型输入尺寸
        resized = cv2.resize(image_rgb, self.input_size)

        # 归一化到 [0, 1]
        normalized = resized.astype('float32') / 255.0

        # 添加批次维度: (1, 256, 256, 3)
        batched = np.expand_dims(normalized, axis=0)

        return batched, original_shape

    def postprocess_image(self, output, original_shape):
        """
        后处理模型输出

        参数:
            output: 模型输出
            original_shape: 原始图像尺寸

        返回:
            处理后的图像 (BGR格式)
        """
        # 移除批次维度
        dehazed = output[0]

        # 裁剪到 [0, 1] 范围
        dehazed = np.clip(dehazed, 0, 1)

        # 调整回原始尺寸
        dehazed_resized = cv2.resize(dehazed, (original_shape[1], original_shape[0]))

        # 转换为8位图像
        dehazed_uint8 = (dehazed_resized * 255).astype('uint8')

        # 转换回BGR格式
        dehazed_bgr = cv2.cvtColor(dehazed_uint8, cv2.COLOR_RGB2BGR)

        return dehazed_bgr

    def dehaze_single(self, input_path, output_path=None):
        """
        对单张图像进行去雾

        参数:
            input_path: 输入图像路径
            output_path: 输出图像路径，如果为None则不保存

        返回:
            去雾后的图像
        """
        print(f"正在处理图像: {input_path}")

        # 检查文件是否存在
        if not os.path.exists(input_path):
            print(f"✗ 错误: 文件不存在 - {input_path}")
            return None

        try:
            # 读取图像
            image = cv2.imread(input_path)
            if image is None:
                print(f"✗ 错误: 无法读取图像 - {input_path}")
                return None

            print(f"  原始尺寸: {image.shape}")

            # 预处理
            processed, original_shape = self.preprocess_image(image)

            # 进行去雾
            print("  正在进行去雾处理...")
            output = self.model.predict(processed, verbose=0)

            # 后处理
            result = self.postprocess_image(output, original_shape)

            # 保存结果
            if output_path:
                cv2.imwrite(output_path, result)
                print(f"✓ 结果已保存到: {output_path}")

            return result

        except Exception as e:
            print(f"✗ 处理图像时出错: {e}")
            return None

    def dehaze_batch(self, input_dir, output_dir):
        """
        批量处理文件夹中的图像

        参数:
            input_dir: 输入文件夹路径
            output_dir: 输出文件夹路径
        """
        print(f"批量处理: {input_dir}")

        # 检查输入目录
        if not os.path.exists(input_dir):
            print(f"✗ 错误: 输入目录不存在 - {input_dir}")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # 获取所有图像文件
        image_files = []
        for file in os.listdir(input_dir):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(file)

        if not image_files:
            print("✗ 错误: 未找到支持的图像文件")
            return

        print(f"  找到 {len(image_files)} 张图像")

        # 处理每张图像
        success_count = 0
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_dir, filename)

            # 生成输出文件名
            name_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{name_without_ext}_dehazed.jpg")

            print(f"  [{i}/{len(image_files)}] 处理: {filename}")

            # 处理单张图像
            result = self.dehaze_single(input_path, output_path)
            if result is not None:
                success_count += 1

        print(f"\n✓ 批量处理完成!")
        print(f"  成功处理: {success_count}/{len(image_files)} 张图像")
        print(f"  结果保存在: {output_dir}")

    def compare_results(self, input_path, dehazed_path, save_path=None):
        """
        比较原始图像和去雾结果

        参数:
            input_path: 原始图像路径
            dehazed_path: 去雾图像路径
            save_path: 对比图保存路径
        """
        # 读取图像
        original = cv2.imread(input_path)
        dehazed = cv2.imread(dehazed_path)

        if original is None or dehazed is None:
            print("✗ 错误: 无法读取图像")
            return

        # 将BGR转换为RGB用于显示
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        dehazed_rgb = cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB)

        # 创建对比图
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title('原始有雾图像')
        plt.imshow(original_rgb)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('去雾结果')
        plt.imshow(dehazed_rgb)
        plt.axis('off')

        plt.tight_layout()

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 对比图已保存到: {save_path}")

        plt.show()

    def test_model(self, test_image_path=None):
        """
        测试模型

        参数:
            test_image_path: 测试图像路径，如果为None则使用默认图像
        """
        print("正在测试模型...")

        # 如果没有提供测试图像，创建一个简单的测试图像
        if test_image_path is None or not os.path.exists(test_image_path):
            print("  创建测试图像...")
            # 创建一个简单的彩色图像作为测试
            test_image = np.zeros((256, 256, 3), dtype=np.uint8)
            test_image[:, :, 0] = 100  # 蓝色通道
            test_image[:, :, 1] = 150  # 绿色通道
            test_image[:, :, 2] = 200  # 红色通道
        else:
            test_image = cv2.imread(test_image_path)

        # 预处理
        processed, original_shape = self.preprocess_image(test_image)

        # 测试预测
        print("  进行预测...")
        start_time = time.time()
        output = self.model.predict(processed, verbose=0)
        inference_time = time.time() - start_time

        print(f"✓ 测试通过!")
        print(f"  推理时间: {inference_time:.3f} 秒")
        print(f"  输入形状: {processed.shape}")
        print(f"  输出形状: {output.shape}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='图像去雾工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 单张图像去雾
  python dehaze_app.py hazy.jpg

  # 指定输出路径
  python dehaze_app.py hazy.jpg dehazed.jpg

  # 批量处理文件夹
  python dehaze_app.py --batch hazy_images/ dehazed_results/

  # 使用不同模型
  python dehaze_app.py --model my_model.keras hazy.jpg

  # 生成对比图
  python dehaze_app.py --compare hazy.jpg dehazed.jpg comparison.jpg

  # 测试模型
  python dehaze_app.py --test
        '''
    )

    parser.add_argument('input', nargs='?', help='输入图像路径或文件夹路径')
    parser.add_argument('output', nargs='?', help='输出图像路径或文件夹路径')
    parser.add_argument('--model', '-m', default='best_generator.keras',
                        help='模型文件路径 (默认: best_generator.keras)')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='批量处理文件夹')
    parser.add_argument('--compare', '-c', action='store_true',
                        help='生成原始和去雾图像的对比图')
    parser.add_argument('--test', '-t', action='store_true',
                        help='测试模型')

    args = parser.parse_args()

    # 显示标题
    print("=" * 60)
    print("图像去雾模型 v1.0")
    print("=" * 60)

    # 创建去雾应用实例
    app = DehazeApp(args.model)

    # 根据参数执行不同操作
    if args.test:
        # 测试模型
        app.test_model()

    elif args.compare:
        # 生成对比图
        if not args.input or not args.output:
            print("✗ 错误: 对比模式需要输入图像和去雾图像路径")
            sys.exit(1)

        save_path = args.output if args.output else 'comparison.jpg'
        app.compare_results(args.input, args.output, save_path)

    elif args.batch:
        # 批量处理
        if not args.input:
            print("✗ 错误: 批量处理需要输入文件夹路径")
            sys.exit(1)

        output_dir = args.output if args.output else f"{args.input}_dehazed"
        app.dehaze_batch(args.input, output_dir)

    elif args.input:
        # 单张图像处理
        if os.path.isdir(args.input):
            print("✗ 错误: 输入是文件夹，请使用 --batch 参数")
            sys.exit(1)

        # 生成输出路径
        if args.output:
            output_path = args.output
        else:
            name, ext = os.path.splitext(args.input)
            output_path = f"{name}_dehazed.jpg"

        # 处理图像
        result = app.dehaze_single(args.input, output_path)

        if result is not None:
            # 询问是否显示对比图
            print("\n是否显示对比图? (y/n): ", end='')
            choice = input().strip().lower()

            if choice == 'y':
                comparison_path = f"{os.path.splitext(output_path)[0]}_comparison.jpg"
                app.compare_results(args.input, output_path, comparison_path)

    else:
        # 交互模式
        print("\n请选择操作:")
        print("  1. 单张图像去雾")
        print("  2. 批量处理文件夹")
        print("  3. 测试模型")
        print("  4. 退出")

        choice = input("请输入选项 (1-4): ").strip()

        if choice == '1':
            input_path = input("请输入输入图像路径: ").strip()
            output_path = input("请输入输出图像路径 (直接回车使用默认): ").strip()

            if not output_path:
                name, ext = os.path.splitext(input_path)
                output_path = f"{name}_dehazed.jpg"

            result = app.dehaze_single(input_path, output_path)

            if result is not None:
                print(f"\n是否显示对比图? (y/n): ", end='')
                if input().strip().lower() == 'y':
                    app.compare_results(input_path, output_path)

        elif choice == '2':
            input_dir = input("请输入输入文件夹路径: ").strip()
            output_dir = input("请输入输出文件夹路径 (直接回车使用默认): ").strip()

            if not output_dir:
                output_dir = f"{input_dir}_dehazed"

            app.dehaze_batch(input_dir, output_dir)

        elif choice == '3':
            app.test_model()

        elif choice == '4':
            print("退出程序")
        else:
            print("无效选项")


if __name__ == "__main__":
    # 导入time模块用于计时
    import time

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n✗ 程序运行出错: {e}")