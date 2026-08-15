import argparse
import gc
import json
import os


def require_directory(
    path_value,
    env_name,
):
    if not path_value:
        raise RuntimeError(
            "请设置环境变量 {}".format(
                env_name
            )
        )

    path = os.path.abspath(
        os.path.expanduser(
            path_value
        )
    )

    if not os.path.isdir(path):
        raise FileNotFoundError(
            "模型目录不存在：{}".format(
                path
            )
        )

    return path


def main_generate(args):
    tool_name = args.get("tool")

    if tool_name == "region_SDXL":
        from region_sd.noise_testxl import generate

        generate(
            args["input"]["prompt1"],
            args["input"]["prompt2"],
            args["input"]["mask"],
            args["output"],
        )

    elif tool_name == "region_SD":
        from region_sd.noise_test import generate

        generate(
            args["input"]["prompt1"],
            args["input"]["prompt2"],
            args["input"]["mask"],
            args["output"],
        )

    elif tool_name == "CreatiLayout":
        model_path = require_directory(
            os.getenv(
                "SD3_MODEL_PATH"
            ),
            "SD3_MODEL_PATH",
        )

        ckpt_path = require_directory(
            os.getenv(
                "CREATILAYOUT_MODEL_PATH"
            ),
            "CREATILAYOUT_MODEL_PATH",
        )

        model_index = os.path.join(
            model_path,
            "model_index.json",
        )

        if not os.path.isfile(
            model_index
        ):
            raise FileNotFoundError(
                "SD3 模型目录中缺少：{}".format(
                    model_index
                )
            )

        transformer_config = (
            os.path.join(
                ckpt_path,
                "transformer",
                "config.json",
            )
        )

        if not os.path.isfile(
            transformer_config
        ):
            raise FileNotFoundError(
                "CreatiLayout 权重目录中缺少：{}".format(
                    transformer_config
                )
            )

        input_data = args.get(
            "input"
        )

        if not isinstance(
            input_data,
            dict,
        ):
            raise ValueError(
                "生成参数中的 input "
                "必须是字典"
            )

        for key in (
            "region1",
            "region2",
            "box",
            "mask",
        ):
            if key not in input_data:
                raise ValueError(
                    "生成参数缺少 input.{}".format(
                        key
                    )
                )

        mask_path = os.path.abspath(
            os.path.expanduser(
                input_data["mask"]
            )
        )

        if not os.path.isfile(
            mask_path
        ):
            raise FileNotFoundError(
                "找不到 mask：{}".format(
                    mask_path
                )
            )

        input_data["mask"] = mask_path

        output_path = args.get(
            "output"
        )

        if not output_path:
            raise ValueError(
                "生成参数缺少 output"
            )

        output_path = os.path.abspath(
            os.path.expanduser(
                output_path
            )
        )

        os.makedirs(
            os.path.dirname(
                output_path
            ),
            exist_ok=True,
        )

        from CreatiLayout.creatilayout_generate import generate

        generate(
            input_data,
            output_path,
            seed=args.get(
                "seed",
                42,
            ),
            c_T=args.get(
                "c_T",
                0,
            ),
            guidance_scale=args.get(
                "guidance_scale",
                5,
            ),
            scale_factor=args.get(
                "scale_factor",
                0,
            ),
            model_path=model_path,
            ckpt_path=ckpt_path,
            num_inference_steps=args.get(
                "num_inference_steps",
                50,
            ),
            change_solver=args.get(
                "change_solver",
                False,
            ),
        )

        if not os.path.isfile(
            output_path
        ):
            raise RuntimeError(
                "CreatiLayout 运行结束，"
                "但没有生成图片：{}".format(
                    output_path
                )
            )

    else:
        raise ValueError(
            "不支持的生成工具：{}".format(
                tool_name
            )
        )

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    finally:
        gc.collect()


def parse_bool(value):
    if isinstance(value, bool):
        return value

    value = str(value).lower()

    if value in (
        "true",
        "1",
        "yes",
        "y",
    ):
        return True

    if value in (
        "false",
        "0",
        "no",
        "n",
    ):
        return False

    raise argparse.ArgumentTypeError(
        "无法识别布尔值：{}".format(
            value
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run image generation tool"
        )
    )

    parser.add_argument(
        "--json_out",
        type=parse_bool,
        default=False,
    )

    parser.add_argument(
        "--json_path",
        default="gen_text.json",
    )

    command_args = (
        parser.parse_args()
    )

    if not command_args.json_out:
        raise RuntimeError(
            "请使用 --json_out True"
        )

    with open(
        command_args.json_path,
        "r",
        encoding="utf-8",
    ) as file:
        request_args = json.load(
            file
        )

    main_generate(
        request_args
    )
