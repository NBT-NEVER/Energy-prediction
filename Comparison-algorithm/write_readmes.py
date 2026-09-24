# _*_coding:UTF-8_*_
# 开发者: NBT
# 文件名: write_readmes.py
# 开发时间: 2026-09-09
# 文件名: write_readmes.py
# 功能说明: 兼容旧命令并调用4.1统一实验文档生成器
# 版本号：4.1


def main() -> None:
    """功能: 调用统一生成器更新总README和各算法README。
    参数: 无。
    返回: 无。
    调用位置: 旧版write_readmes.py命令入口。
    """

    from generate_documentation import main as generate_documentation  # 延迟加载文档生成入口

    generate_documentation()


if __name__ == "__main__":
    main()
