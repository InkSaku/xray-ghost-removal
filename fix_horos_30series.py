from pathlib import Path
import argparse
import hashlib
import re
import shutil

import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import (
    UID,
    generate_uid,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    ExplicitVRBigEndian,
    SecondaryCaptureImageStorage,
)


# ============================================================
# 1. 基础配置
# ============================================================

DICOM_EXTENSIONS = {
    ".dcm",
    ".dicom",
}


# ============================================================
# 2. 自然排序
# ============================================================

def natural_key(text):
    """
    自然排序：

    1.dcm
    2.dcm
    3.dcm
    ...
    10.dcm
    """

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(text))
    ]


# ============================================================
# 3. UTF-8 安全截断
# ============================================================

def safe_text(text, max_bytes=64):

    text = str(text)

    encoded = text.encode("utf-8")

    if len(encoded) <= max_bytes:
        return text

    encoded = encoded[:max_bytes]

    while encoded:

        try:
            return encoded.decode("utf-8")

        except UnicodeDecodeError:
            encoded = encoded[:-1]

    return ""


# ============================================================
# 4. PixelData SHA256
# ============================================================

def pixel_hash(ds):

    if "PixelData" not in ds:
        return None

    return hashlib.sha256(
        ds.PixelData
    ).hexdigest()


# ============================================================
# 5. File Meta
# ============================================================

def ensure_file_meta(ds):

    if (
        not hasattr(ds, "file_meta")
        or ds.file_meta is None
    ):
        ds.file_meta = FileMetaDataset()


# ============================================================
# 6. 判断 PixelData 是否为 encapsulated
# ============================================================

def looks_encapsulated(ds):

    if "PixelData" not in ds:
        return False

    data = ds.PixelData

    return (
        len(data) >= 4
        and data[:4] == b"\xfe\xff\x00\xe0"
    )


# ============================================================
# 7. Transfer Syntax
# ============================================================

def ensure_transfer_syntax(ds, filename):

    ensure_file_meta(ds)

    # 原来有就不动
    if "TransferSyntaxUID" in ds.file_meta:
        return

    # 压缩数据但缺 Transfer Syntax，不能乱猜
    if looks_encapsulated(ds):

        raise RuntimeError(
            f"\n{filename}\n"
            "PixelData 看起来是压缩数据，"
            "但 TransferSyntaxUID 缺失。\n"
            "为避免损坏数据，停止处理。"
        )

    is_little_endian = getattr(
        ds,
        "is_little_endian",
        True,
    )

    is_implicit_vr = getattr(
        ds,
        "is_implicit_VR",
        False,
    )

    if is_little_endian is False:

        ds.file_meta.TransferSyntaxUID = (
            ExplicitVRBigEndian
        )

    elif is_implicit_vr:

        ds.file_meta.TransferSyntaxUID = (
            ImplicitVRLittleEndian
        )

    else:

        ds.file_meta.TransferSyntaxUID = (
            ExplicitVRLittleEndian
        )


# ============================================================
# 8. SOP Class
# ============================================================

def ensure_sop_class(ds):

    ensure_file_meta(ds)

    sop_class_uid = ds.get(
        "SOPClassUID",
        None,
    )

    if sop_class_uid is None:

        sop_class_uid = ds.file_meta.get(
            "MediaStorageSOPClassUID",
            None,
        )

    if sop_class_uid is None:

        sop_class_uid = (
            SecondaryCaptureImageStorage
        )

        print(
            "  ⚠ 缺少 SOPClassUID，"
            "使用 Secondary Capture"
        )

    ds.SOPClassUID = sop_class_uid

    ds.file_meta.MediaStorageSOPClassUID = (
        sop_class_uid
    )


# ============================================================
# 9. NumberOfFrames 安全处理
# ============================================================

def validate_number_of_frames(ds, filename):
    """
    当前数据原则：

        一个 DICOM 文件应对应一张二维图像。

    如果 Header 错误声明多帧，
    但 PixelData 实际只有一帧，
    删除 NumberOfFrames。

    如果确实是真正多帧，则保留。
    """

    if "NumberOfFrames" not in ds:

        return "absent"

    original_value = ds.NumberOfFrames

    try:

        number_of_frames = int(
            original_value
        )

    except Exception:

        print(
            f"  ⚠ NumberOfFrames={original_value} "
            "无法解析，保持原样"
        )

        return "unparseable-kept"


    # --------------------------------------------------------
    # 正常单帧
    # --------------------------------------------------------

    if number_of_frames == 1:

        return "single-kept"


    # --------------------------------------------------------
    # 非法值
    # --------------------------------------------------------

    if number_of_frames <= 0:

        del ds.NumberOfFrames

        print(
            f"  ⚠ NumberOfFrames={number_of_frames}，"
            "已删除"
        )

        return "invalid-removed"


    print(
        f"  Header NumberOfFrames="
        f"{number_of_frames}"
    )


    # --------------------------------------------------------
    # 获取 Transfer Syntax
    # --------------------------------------------------------

    transfer_syntax = None

    if (
        hasattr(ds, "file_meta")
        and ds.file_meta is not None
    ):

        transfer_syntax = ds.file_meta.get(
            "TransferSyntaxUID",
            None,
        )


    is_compressed = False

    if transfer_syntax is not None:

        try:

            is_compressed = UID(
                str(transfer_syntax)
            ).is_compressed

        except Exception:

            pass


    # --------------------------------------------------------
    # 未压缩数据：根据 PixelData 大小判断
    # --------------------------------------------------------

    if (
        not is_compressed
        and not looks_encapsulated(ds)
    ):

        try:

            rows = int(
                ds.get("Rows", 0)
            )

            columns = int(
                ds.get("Columns", 0)
            )

            samples_per_pixel = int(
                ds.get("SamplesPerPixel", 1)
            )

            bits_allocated = int(
                ds.get("BitsAllocated", 0)
            )

        except Exception:

            rows = 0
            columns = 0
            samples_per_pixel = 0
            bits_allocated = 0


        if (
            rows > 0
            and columns > 0
            and samples_per_pixel > 0
            and bits_allocated > 0
            and "PixelData" in ds
        ):

            frame_bits = (
                rows
                * columns
                * samples_per_pixel
                * bits_allocated
            )

            frame_bytes = (
                frame_bits + 7
            ) // 8

            pixel_bytes = len(
                ds.PixelData
            )


            # 实际只有一帧
            if pixel_bytes in {
                frame_bytes,
                frame_bytes + 1,
            }:

                print(
                    "  ⚠ Header 声称多帧，"
                    "但 PixelData 实际只有一帧"
                )

                print(
                    "  → 删除错误 NumberOfFrames"
                )

                del ds.NumberOfFrames

                return "false-multiframe-fixed"


            # 真多帧
            expected_bytes = (
                frame_bytes
                * number_of_frames
            )

            if pixel_bytes in {
                expected_bytes,
                expected_bytes + 1,
            }:

                print(
                    f"  ✓ 确认真正 {number_of_frames} 帧"
                )

                return "multiframe-kept"


    # --------------------------------------------------------
    # 再尝试解码确认
    # --------------------------------------------------------

    try:

        pixel_array = ds.pixel_array

        samples_per_pixel = int(
            ds.get(
                "SamplesPerPixel",
                1,
            )
        )


        # 灰度单帧
        if (
            samples_per_pixel == 1
            and pixel_array.ndim == 2
        ):

            print(
                "  ⚠ 实际解码为二维单帧，"
                "删除错误 NumberOfFrames"
            )

            del ds.NumberOfFrames

            return "false-multiframe-fixed"


        # 灰度多帧
        if (
            samples_per_pixel == 1
            and pixel_array.ndim == 3
            and pixel_array.shape[0]
            == number_of_frames
        ):

            return "multiframe-kept"


        # 彩色单帧
        if (
            samples_per_pixel > 1
            and pixel_array.ndim == 3
            and pixel_array.shape[-1]
            == samples_per_pixel
        ):

            del ds.NumberOfFrames

            return "false-multiframe-fixed"


        # 彩色多帧
        if (
            samples_per_pixel > 1
            and pixel_array.ndim == 4
            and pixel_array.shape[0]
            == number_of_frames
        ):

            return "multiframe-kept"


        print(
            "  ⚠ 无法确认帧结构，保持原样：",
            pixel_array.shape,
        )

        return "ambiguous-kept"


    except Exception as e:

        print(
            "  ⚠ 无法解码进一步检查：",
            e,
        )

        return "decode-failed-kept"


# ============================================================
# 10. 获取 DICOM 文件
# ============================================================

def get_dicom_files(folder):

    files = []

    for path in folder.iterdir():

        if not path.is_file():
            continue

        if path.name.startswith("."):
            continue

        if (
            path.suffix.lower()
            not in DICOM_EXTENSIONS
        ):
            continue

        files.append(
            path
        )

    return sorted(
        files,
        key=lambda p: natural_key(
            p.name
        ),
    )


# ============================================================
# 11. 主处理函数
# ============================================================

def process_dataset(
    input_dir,
    output_dir,
    overwrite=False,
):

    input_dir = (
        input_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        output_dir
        .expanduser()
        .resolve()
    )


    # ========================================================
    # 输入检查
    # ========================================================

    if not input_dir.exists():

        raise FileNotFoundError(
            f"\n输入目录不存在：\n"
            f"{input_dir}"
        )


    if not input_dir.is_dir():

        raise NotADirectoryError(
            f"\n输入路径不是文件夹：\n"
            f"{input_dir}"
        )


    # ========================================================
    # 防止输出放进输入
    # ========================================================

    try:

        output_dir.relative_to(
            input_dir
        )

        raise RuntimeError(
            "\n输出目录不能位于输入目录内部。"
        )

    except ValueError:

        pass


    # ========================================================
    # 输出目录
    # ========================================================

    if output_dir.exists():

        if not overwrite:

            raise FileExistsError(
                "\n输出目录已经存在：\n"
                f"{output_dir}\n\n"
                "重新生成请添加 --overwrite"
            )

        print(
            "删除旧输出目录：",
            output_dir,
        )

        shutil.rmtree(
            output_dir
        )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # 获取所有 DICOM
    # ========================================================

    files = get_dicom_files(
        input_dir
    )


    if not files:

        raise RuntimeError(
            "\n没有找到任何 DICOM 文件。"
        )


    # ========================================================
    # 所有 30 张共用一个 Study
    # ========================================================

    study_uid = generate_uid()

    dataset_name = input_dir.name


    print()
    print("=" * 80)

    print(
        "DICOM → Horos：每张图独立 Series"
    )

    print("=" * 80)

    print(
        "输入目录：",
        input_dir,
    )

    print(
        "输出目录：",
        output_dir,
    )

    print(
        "图片数量：",
        len(files),
    )

    print()


    success_count = 0


    # ========================================================
    # 每一张 DICOM = 一个 Series
    # ========================================================

    for series_number, input_path in enumerate(
        files,
        start=1,
    ):

        filename = (
            input_path.name
        )

        stem = (
            input_path.stem
        )


        print()
        print(
            f"[{series_number}/{len(files)}] "
            f"{filename}"
        )


        # ----------------------------------------------------
        # 读取
        # ----------------------------------------------------

        ds = pydicom.dcmread(
            str(input_path),
            force=True,
        )


        if "PixelData" not in ds:

            print(
                "  ✗ 没有 PixelData，跳过"
            )

            continue


        # ----------------------------------------------------
        # 保存原始信息
        # ----------------------------------------------------

        old_pixel_hash = pixel_hash(
            ds
        )

        old_rows = ds.get(
            "Rows",
            None,
        )

        old_columns = ds.get(
            "Columns",
            None,
        )

        old_bits_allocated = ds.get(
            "BitsAllocated",
            None,
        )

        old_bits_stored = ds.get(
            "BitsStored",
            None,
        )


        # ----------------------------------------------------
        # File Meta
        # ----------------------------------------------------

        ensure_file_meta(
            ds
        )

        ensure_transfer_syntax(
            ds,
            filename,
        )

        ensure_sop_class(
            ds
        )


        # ----------------------------------------------------
        # NumberOfFrames
        # ----------------------------------------------------

        frame_action = (
            validate_number_of_frames(
                ds,
                filename,
            )
        )


        # ====================================================
        # Patient
        #
        # 30 张仍属于同一个 Patient
        # ====================================================

        ds.SpecificCharacterSet = (
            "ISO_IR 192"
        )

        ds.PatientName = (
            safe_text(
                dataset_name
            )
        )

        ds.PatientID = (
            safe_text(
                dataset_name
            )
        )


        # ====================================================
        # Study
        #
        # 30 张属于同一个 Study
        # ====================================================

        ds.StudyInstanceUID = (
            study_uid
        )

        ds.StudyID = "1"

        ds.StudyDescription = (
            safe_text(
                dataset_name
            )
        )


        # ====================================================
        # Series
        #
        # 核心：
        # 每张 DICOM 独立一个 Series
        # ====================================================

        new_series_uid = (
            generate_uid()
        )

        ds.SeriesInstanceUID = (
            new_series_uid
        )

        ds.SeriesNumber = (
            series_number
        )


        # ====================================================
        # Series 名
        #
        # 3.dcm -> 03_3.dcm
        # 4.dcm -> 04_4.dcm
        #
        # Horos 首页直接显示这个名字。
        # ====================================================

        series_name = (
            f"{series_number:02d}_{filename}"
        )

        ds.SeriesDescription = (
            safe_text(
                series_name
            )
        )


        # ProtocolName 有些 DICOM 浏览器也会读取
        ds.ProtocolName = (
            safe_text(
                series_name
            )
        )


        # ====================================================
        # 每个 Series 只有一张图
        # ====================================================

        ds.InstanceNumber = 1


        # ====================================================
        # 保存原始文件名
        # ====================================================

        ds.ContentDescription = (
            safe_text(
                filename
            )
        )

        ds.ImageComments = (
            safe_text(
                f"Original file: {filename}",
                max_bytes=1024,
            )
        )


        # ====================================================
        # 每张 DICOM 唯一 SOP Instance UID
        # ====================================================

        new_sop_uid = (
            generate_uid()
        )

        ds.SOPInstanceUID = (
            new_sop_uid
        )

        ds.file_meta.MediaStorageSOPInstanceUID = (
            new_sop_uid
        )


        # ====================================================
        # 输出
        #
        # 实际文件名仍保持原样：
        #
        # 3.dcm 还是 3.dcm
        # ====================================================

        output_path = (
            output_dir
            / filename
        )


        ds.save_as(
            str(output_path),
            enforce_file_format=True,
        )


        # ====================================================
        # 重新读取验证
        # ====================================================

        new_ds = pydicom.dcmread(
            str(output_path),
            force=True,
        )


        # ----------------------------------------------------
        # PixelData 必须完全没变
        # ----------------------------------------------------

        new_pixel_hash = pixel_hash(
            new_ds
        )


        if (
            old_pixel_hash
            !=
            new_pixel_hash
        ):

            raise RuntimeError(
                f"\n严重错误：{filename}\n"
                "PixelData 发生变化！"
            )


        # ----------------------------------------------------
        # 基础图像参数检查
        # ----------------------------------------------------

        checks = {
            "Rows":
                old_rows,

            "Columns":
                old_columns,

            "BitsAllocated":
                old_bits_allocated,

            "BitsStored":
                old_bits_stored,
        }


        for tag_name, old_value in (
            checks.items()
        ):

            new_value = new_ds.get(
                tag_name,
                None,
            )

            if new_value != old_value:

                raise RuntimeError(
                    f"\n{filename}\n"
                    f"{tag_name} 发生变化：\n"
                    f"{old_value} -> {new_value}"
                )


        success_count += 1


        # ====================================================
        # 输出结果
        # ====================================================

        print(
            "  ✓ Horos Series 名：",
            series_name,
        )

        print(
            "  ✓ SeriesNumber：",
            series_number,
        )

        print(
            "  ✓ 原始文件：",
            filename,
        )

        print(
            "  ✓ NumberOfFrames：",
            frame_action,
        )

        print(
            "  ✓ PixelData：未改变"
        )


    # ========================================================
    # 最终检查
    # ========================================================

    output_files = get_dicom_files(
        output_dir
    )


    study_uids = set()

    series_uids = set()

    sop_uids = set()

    series_names = []


    for path in output_files:

        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
        )

        study_uids.add(
            str(
                ds.StudyInstanceUID
            )
        )

        series_uids.add(
            str(
                ds.SeriesInstanceUID
            )
        )

        sop_uids.add(
            str(
                ds.SOPInstanceUID
            )
        )

        series_names.append(
            str(
                ds.SeriesDescription
            )
        )


    # ========================================================
    # 最终必须满足：
    #
    # 1 Study
    # 30 Series
    # 30 SOP
    # ========================================================

    if len(study_uids) != 1:

        raise RuntimeError(
            "错误：不是 1 个 Study。"
        )


    if (
        len(series_uids)
        !=
        len(output_files)
    ):

        raise RuntimeError(
            "错误：SeriesInstanceUID "
            "没有做到每张唯一。"
        )


    if (
        len(sop_uids)
        !=
        len(output_files)
    ):

        raise RuntimeError(
            "错误：SOPInstanceUID "
            "存在重复。"
        )


    # ========================================================
    # 完成
    # ========================================================

    print()
    print()
    print("=" * 80)

    print(
        "全部处理完成"
    )

    print("=" * 80)

    print(
        "成功处理：",
        success_count,
    )

    print(
        "Study 数量：",
        len(study_uids),
    )

    print(
        "Series 数量：",
        len(series_uids),
    )

    print(
        "SOP 数量：",
        len(sop_uids),
    )

    print()

    print(
        "Horos 中应该显示："
    )

    for name in series_names:

        print(
            f"  {name}    1 Image"
        )

    print()

    print(
        "输出目录："
    )

    print(
        output_dir
    )

    print()

    print(
        "✓ PixelData 未修改"
    )

    print(
        "✓ 30 张属于同一个 Study"
    )

    print(
        "✓ 每张图独立一个 Series"
    )

    print(
        "✓ Series 名直接包含原始文件名"
    )


# ============================================================
# 12. 命令行入口
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "将一个目录中的 DICOM "
            "整理成每张图一个 Horos Series，"
            "并使用原始文件名作为 Series 名。"
        )
    )


    parser.add_argument(
        "input",
        type=Path,
        help=(
            "原始 DICOM 目录"
        ),
    )


    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "输出目录"
        ),
    )


    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "删除已有输出并重新生成"
        ),
    )


    args = parser.parse_args()


    input_dir = (
        args.input
    )


    if args.output is None:

        output_dir = (
            input_dir.parent
            / (
                input_dir.name
                + "_horos_30series"
            )
        )

    else:

        output_dir = (
            args.output
        )


    process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )


# ============================================================
# 13. 程序入口
# ============================================================

if __name__ == "__main__":

    main()