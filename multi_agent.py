from util import Agent, add_text_blocks
from util.poster_text import (
    build_agent_prompt,
    extract_theme_and_visual_texts,
    normalize_text_layout,
    parse_structured_response,
    split_poster_design_for_generation,
)
import json
import os
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class multi_agent:
    def __init__(
        self,
        api,
        url,
        system_messages,
        img_dir_path,
        final_img_path,
        mask_path,
        user_prompt,
        max_times=3,
        use_history=True,
    ):
        self.api = api
        self.url = url
        self.system_messages = system_messages
        self.use_history = use_history

        self.element_generator = Agent(
            api,
            url,
            system_messages["element_generator"],
            use_history,
        )
        self.layout_generator = Agent(
            api,
            url,
            system_messages["layout_generator"],
            use_history,
        )
        self.critictor = Agent(
            api,
            url,
            system_messages["critictor"],
            use_history,
        )

        self.img_num = 0
        self.gen_history = []
        self.chat_history = []
        self.mask_path = mask_path

        os.makedirs(img_dir_path, exist_ok=True)

        self.img_dir_path = img_dir_path
        self.final_img_path = final_img_path
        self.max_times = max_times
        self.regen_time = 0
        self.user_prompt = user_prompt
        self.theme = user_prompt
        self.visual_texts = []
        self.current_text_layout = []

        self.timing_stats = {
            "total_time": 0,
            "preprocess_time": 0,
            "agent_time": {
                "element_generator": 0,
                "layout_generator": 0,
                "critictor": 0,
            },
            "agent_num": {
                "element_generator": 0,
                "layout_generator": 0,
                "critictor": 0,
            },
            "tool_generation_time": 0,
            "tool_generation_num": 0,
            "text_render_time": 0,
            "start_time": None,
            "end_time": None,
            "detailed_logs": [],
        }

    def gen(self):
        if not os.path.isfile(self.mask_path):
            raise FileNotFoundError(
                "找不到区域 mask：{}".format(
                    self.mask_path
                )
            )

        print("Start generating poster")

        self.timing_stats["start_time"] = time.time()

        self._extract_text_inputs()

        self.element_gen_chat(
            build_agent_prompt(
                self.theme,
                self.visual_texts,
            ),
            role="user",
            images_url=[self.mask_path],
        )

        self.timing_stats["end_time"] = time.time()

        self.timing_stats["total_time"] = (
            self.timing_stats["end_time"]
            - self.timing_stats["start_time"]
        )

    def _extract_text_inputs(self):
        start_time = time.time()

        try:
            info, response = extract_theme_and_visual_texts(
                self.api,
                self.url,
                self.user_prompt,
                self.system_messages.get(
                    "text_extractor"
                ),
            )

            self.theme = info["theme"]
            self.visual_texts = info["visual_texts"]

            self.chat_history.append(
                {
                    "stage": "text_extractor_preprocess",
                    "content": response,
                }
            )

            print("Text preprocessing completed")

            print(
                json.dumps(
                    info,
                    ensure_ascii=False,
                    indent=4,
                )
            )

        except Exception as error:
            print(
                "Text preprocessing failed, "
                "use original prompt as theme: {}".format(
                    error
                )
            )

            info, _ = extract_theme_and_visual_texts(
                None,
                None,
                self.user_prompt,
                None,
            )

            self.theme = info["theme"]
            self.visual_texts = info["visual_texts"]

        self.timing_stats["preprocess_time"] += (
            time.time() - start_time
        )

    def element_gen_chat(
        self,
        text,
        role="user",
        images_url=None,
    ):
        print("element_generator is working")

        start_time = time.time()

        response = self.element_generator.chat(
            text,
            role=role,
            images_url=images_url,
        )

        self.timing_stats["agent_time"][
            "element_generator"
        ] += time.time() - start_time

        self.timing_stats["agent_num"][
            "element_generator"
        ] += 1

        self.chat_history.append(
            {
                "agent": "element_generator",
                "content": response,
            }
        )

        print(response)

        input_text = "Elements Input:" + response

        if self.img_num != 0:
            input_text = (
                "Critictor think that the previous "
                "elements is not good, you need to "
                "generate new layout for new elements.\n"
                + input_text
            )

        self.layout_gen_chat(
            build_agent_prompt(
                self.theme,
                self.visual_texts,
                input_text,
            ),
            role=role,
            images_url=images_url,
        )

    def layout_gen_chat(
        self,
        text,
        role="user",
        images_url=None,
    ):
        print("layout_generator is working")

        start_time = time.time()

        response = self.layout_generator.chat(
            text,
            role=role,
            images_url=images_url,
        )

        self.timing_stats["agent_time"][
            "layout_generator"
        ] += time.time() - start_time

        self.timing_stats["agent_num"][
            "layout_generator"
        ] += 1

        self.chat_history.append(
            {
                "agent": "layout_generator",
                "content": response,
            }
        )

        print(response)

        gen_command = parse_structured_response(
            response
        )

        if (
            gen_command["element_regen"]
            and self.regen_time < self.max_times
        ):
            self.regen_time += 1

            input_text = (
                "layout generator think that the "
                "elements need edit or add, "
                "regenerate elements.\n"
                + gen_command["regen_request"]
            )

            self.element_gen_chat(
                build_agent_prompt(
                    self.theme,
                    self.visual_texts,
                    input_text,
                ),
                role=role,
            )

            return

        poster_design = gen_command[
            "poster_design"
        ]

        tool_design, text_layout = (
            split_poster_design_for_generation(
                poster_design
            )
        )

        self.current_text_layout = (
            normalize_text_layout(
                gen_command.get("text_layout")
                or text_layout,
                self.visual_texts,
            )
        )

        tool_design["mask"] = self.mask_path

        img_path = os.path.join(
            self.img_dir_path,
            "img_{}.png".format(
                self.img_num
            ),
        )

        seq_args = {
            "tool": "CreatiLayout",
            "input": tool_design,
            "output": img_path,
        }

        self.gen_history.append(
            {
                "seq_args": seq_args,
                "text_layout":
                    self.current_text_layout,
            }
        )

        gen_text_path = os.path.join(
            PROJECT_ROOT,
            "gen_text.json",
        )

        with open(
            gen_text_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                seq_args,
                file,
                ensure_ascii=False,
                indent=4,
            )

        tool_start_time = time.time()

        subprocess.run(
            [
                sys.executable,
                os.path.join(
                    PROJECT_ROOT,
                    "tool_generate.py",
                ),
                "--json_out",
                "True",
                "--json_path",
                gen_text_path,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        self.timing_stats[
            "tool_generation_time"
        ] += time.time() - tool_start_time

        self.timing_stats[
            "tool_generation_num"
        ] += 1

        if not os.path.isfile(img_path):
            raise FileNotFoundError(
                "图片生成失败，找不到输出文件：{}".format(
                    img_path
                )
            )

        print(
            "Poster {} generated at {}".format(
                self.img_num,
                img_path,
            )
        )

        if self.img_num == 0:
            critic_images = [
                self.mask_path,
                img_path,
            ]
        else:
            critic_images = [img_path]

        self.img_num += 1

        self.critictor_chat(
            "This is the poster image.",
            role=role,
            images_url=critic_images,
        )

    def critictor_chat(
        self,
        text,
        role="user",
        images_url=None,
    ):
        start_time = time.time()

        response = self.critictor.chat(
            text,
            role=role,
            images_url=images_url,
        )

        self.timing_stats["agent_time"][
            "critictor"
        ] += time.time() - start_time

        self.timing_stats["agent_num"][
            "critictor"
        ] += 1

        self.chat_history.append(
            {
                "agent": "critictor",
                "content": response,
            }
        )

        print(response)

        advice = parse_structured_response(
            response
        )

        if (
            advice["need_edit"]
            and self.regen_time < self.max_times
        ):
            self.regen_time += 1

            suggestions = advice[
                "modification_suggestions"
            ]

            if advice["agent"] == "layout_regen":
                print(
                    "Critictor think that the poster "
                    "need edit, regenerate layout."
                )

                input_text = (
                    "Critictor think that the poster "
                    "need edit, regenerate layout.\n"
                    + suggestions
                )

                self.layout_gen_chat(
                    build_agent_prompt(
                        self.theme,
                        self.visual_texts,
                        input_text,
                    ),
                    role=role,
                )

                return

            if advice["agent"] == "element_regen":
                print(
                    "Critictor think that the elements "
                    "need edit, regenerate elements."
                )

                input_text = (
                    "Critictor think that the elements "
                    "need edit, regenerate elements.\n"
                    + suggestions
                )

                self.element_gen_chat(
                    build_agent_prompt(
                        self.theme,
                        self.visual_texts,
                        input_text,
                    ),
                    role=role,
                )

                return

            raise ValueError(
                "Critictor requested regeneration "
                "but returned unknown agent: {}".format(
                    advice["agent"]
                )
            )

        self._finalize_with_text(
            images_url[-1]
        )

    def _finalize_with_text(
        self,
        image_path,
    ):
        print(
            "image has been generated, "
            "render final visual texts"
        )

        output_dir = os.path.dirname(
            self.final_img_path
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        if self.visual_texts:
            start_time = time.time()

            add_text_blocks(
                image_path,
                self.current_text_layout,
                self.final_img_path,
            )

            self.timing_stats[
                "text_render_time"
            ] += time.time() - start_time

        else:
            shutil.copy2(
                image_path,
                self.final_img_path,
            )


if __name__ == "__main__":
    api = os.getenv(
        "RECONTRASTER_API_KEY"
    )

    if not api:
        raise RuntimeError(
            "请先设置环境变量 "
            "RECONTRASTER_API_KEY"
        )

    url = os.getenv(
        "RECONTRASTER_BASE_URL",
        "https://api.openai.com/v1",
    )

    img_dir_path = os.getenv(
        "RECONTRASTER_OUTPUT_DIR",
        os.path.join(
            PROJECT_ROOT,
            "outputs",
        ),
    )

    final_img_path = os.getenv(
        "RECONTRASTER_FINAL_PATH",
        os.path.join(
            PROJECT_ROOT,
            "outputs",
            "final.png",
        ),
    )

    mask_path = os.getenv(
        "RECONTRASTER_MASK_PATH",
        os.path.join(
            PROJECT_ROOT,
            "inputs",
            "mask1.png",
        ),
    )

    prompt = os.getenv(
        "RECONTRASTER_PROMPT",
        "Design a contrast poster calling "
        "for saving water",
    )

    max_times = int(
        os.getenv(
            "RECONTRASTER_MAX_TIMES",
            "3",
        )
    )

    template_files = {
        "critictor":
            "critictor.txt",
        "element_generator":
            "elements_generator.txt",
        "layout_generator":
            "layout_generator.txt",
        "text_extractor":
            "text_extractor.txt",
    }

    system_messages = {}

    for key, filename in (
        template_files.items()
    ):
        path = os.path.join(
            PROJECT_ROOT,
            "templates",
            filename,
        )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            system_messages[key] = (
                file.read()
            )

    pipeline = multi_agent(
        api,
        url,
        system_messages,
        img_dir_path,
        final_img_path,
        mask_path,
        prompt,
        max_times,
        True,
    )

    pipeline.gen()

    print(
        json.dumps(
            pipeline.timing_stats,
            ensure_ascii=False,
            indent=4,
        )
    )
