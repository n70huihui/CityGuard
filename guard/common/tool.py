import os
from pathlib import Path
from PIL import Image


def create_folder_structure(type_name: str, num: int, base_path: str = ".") -> None:
    """
    创建层级文件夹结构：
    1. 在 base_path 下创建 type_name 文件夹
    2. 在 type_name 下创建 num 个子文件夹 (1, 2, ..., num)
    3. 每个子文件夹内创建 cameras 和 monitor 空文件夹
    :param type_name: 主文件夹名称
    :param num: 子文件夹数量
    :param base_path: 基础路径 (默认当前目录)
    :return: 无
    """
    base_dir = Path(base_path) / type_name
    base_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, num + 1):
        sub_dir = base_dir / str(i)
        sub_dir.mkdir(exist_ok=True)

        # 创建子目录中的空文件夹
        (sub_dir / "cameras").mkdir(exist_ok=True)
        (sub_dir / "monitor").mkdir(exist_ok=True)


def compress_png_to_jpg(png_path: str, quality: int = 85) -> None:
    """
    将PNG图片压缩为 JPG 格式并替换原文件
    :param png_path: PNG 图片路径
    :param quality: 压缩质量 (1-100)，默认 85
    :return: 无
    """
    png_path = Path(png_path)
    if png_path.suffix.lower() != '.png':
        raise ValueError("文件必须是PNG格式")

    # 打开图片并转换为RGB模式（去除透明通道）
    img = Image.open(png_path).convert('RGB')

    # 创建JPG路径（相同路径，扩展名改为.jpg）
    jpg_path = png_path.with_suffix('.jpg')

    # 保存为JPG格式
    img.save(jpg_path, 'JPEG', quality=quality)

    # 删除原始PNG文件
    png_path.unlink()


def compress_pngs_in_folder(folder_path: str, quality: int = 85) -> None:
    """
    递归压缩文件夹中所有 PNG 图片为 JPG 格式并替换
    :param folder_path: 文件夹路径
    :param quality: 压缩质量 (1-100)，默认 85
    :return: 无
    """
    folder = Path(folder_path)
    for file_path in folder.rglob('*.png'):
        try:
            compress_png_to_jpg(str(file_path), quality)
            print(f"已压缩: {file_path} -> {file_path.with_suffix('.jpg')}")
        except Exception as e:
            print(f"处理 {file_path} 时出错: {str(e)}")

if __name__ == "__main__":
    compress_pngs_in_folder("C:\MyData\Projects\CodeSpace\Pycharm\CityGuard\datasets\\noise")
    # create_folder_structure(type_name="water", num=10)