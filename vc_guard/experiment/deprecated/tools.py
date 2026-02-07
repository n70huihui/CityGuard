import os
from pathlib import Path
from PIL import Image
import logging


def compress_png_to_jpg(folder_path: str, quality: int = 85):
    """
    递归压缩文件夹中的所有 PNG 图片为 JPG 格式，并删除原始 PNG 文件
    :param folder_path: 要处理的文件夹路径
    :param quality: JPG压缩质量 (0-100)，默认为85
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 转换文件夹路径为Path对象
    folder = Path(folder_path)

    # 验证文件夹是否存在
    if not folder.exists():
        logger.error(f"文件夹不存在: {folder_path}")
        return False

    # 计数器
    processed = 0
    skipped = 0
    errors = 0

    # 递归遍历所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 只处理PNG文件
            if file.lower().endswith('.png'):
                png_path = Path(root) / file
                # 创建对应的JPG路径
                jpg_path = png_path.with_suffix('.jpg')

                try:
                    # 打开PNG图片
                    with Image.open(png_path) as img:
                        # 转换为RGB模式（JPG不支持透明通道）
                        if img.mode in ('RGBA', 'LA'):
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else img)
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')

                        # 保存为JPG格式
                        img.save(jpg_path, 'JPEG', quality=quality)

                    # 删除原始PNG文件
                    png_path.unlink()
                    logger.info(f"成功转换: {png_path} → {jpg_path}")
                    processed += 1

                except Exception as e:
                    logger.error(f"处理文件 {png_path} 时出错: {str(e)}")
                    errors += 1
                    continue

    # 结果统计
    logger.info("\n===== 转换完成 =====")
    logger.info(f"已处理文件: {processed} 个")
    logger.info(f"跳过文件: {skipped} 个")
    logger.info(f"错误文件: {errors} 个")

    return True


# 使用示例
if __name__ == "__main__":
    # 指定要处理的文件夹路径
    target_folder = "C:\MyData\Projects\CodeSpace\Pycharm\CityGuard\datasets\monitor"

    # 调用函数进行转换（质量设置为90）
    compress_png_to_jpg(target_folder, quality=80)
