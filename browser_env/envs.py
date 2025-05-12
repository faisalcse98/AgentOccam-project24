import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import numpy.typing as npt
from beartype import beartype
from beartype.door import is_bearable
from gymnasium import Env
from gymnasium.spaces import Box, Text
from playwright.async_api import (
    CDPSession,
    Page,
    ViewportSize,
    expect,
)

from .actions import Action, execute_action, get_action_space
from .processors import ObservationHandler, ObservationMetadata
from .utils import (
    AccessibilityTree,
    DetachedPage,
    Observation,
    png_bytes_to_numpy,
)

import base64
from .scripts import *

@dataclass
class PlaywrightScript:
    function: str  # goto, get_by_role
    destination: str  # https://www.google.com/, combobox
    name: str | None = None  # Search, Avatar 2009
    operation: str | None = None  # click, fill, press
    value: str | None = None  # avatar movie, Enter


def parse_action(action: str) -> PlaywrightScript:
    splitted = action.strip().split(" ")
    assert len(splitted) >= 2
    match splitted[:2]:
        case ["goto", url]:
            assert len(splitted) == 2
            return PlaywrightScript("goto", url)
        case ["get_by_role", destination]:
            assert len(splitted) >= 4
            match splitted[2:]:
                case [name, operation]:
                    return PlaywrightScript(
                        "get_by_role", destination, name, operation
                    )
                case [name, operation, value]:
                    return PlaywrightScript(
                        "get_by_role", destination, name, operation, value
                    )
                case _:
                    raise ValueError("Invalid action")
        case _:
            raise ValueError(f"Invalid action {action}")


class ScriptBrowserEnv(Env[dict[str, Observation], Action]):
    """
    The goal of this environment is to produce a prototype of a browser environment.
    In the end, we want to support a fully configurable browser environment with wide
    range of action spaces and observation spaces, both structured and unstructured.
    But in this prototype, we just support action space specified by Playwright script,
    and observation space is the html content of the page.
    """

    @beartype
    def __init__(
        self,
        max_page_length: int = 8192,
        headless: bool = True,
        slow_mo: int = 0,
        observation_type: str = "html",
        current_viewport_only: bool = False,
        viewport_size: ViewportSize = {"width": 1280, "height": 720},
        save_trace_enabled: bool = False,
        sleep_after_execution: float = 5.0,
        global_config = None,
        playwright=None,
        page=None,
        context=None,
        context_manager=None,
    ):
        # TODO: make Space[Action] = ActionSpace
        self.action_space = get_action_space()  # type: ignore[assignment]
        self.headless = headless
        self.slow_mo = slow_mo
        self.current_viewport_only = current_viewport_only
        self.reset_finished = False
        self.viewport_size = viewport_size
        self.save_trace_enabled = save_trace_enabled
        self.sleep_after_execution = sleep_after_execution
        self.global_config = global_config
        self.playwright = playwright
        self.page = page
        self.context = context
        self.context_manager = context_manager

        match observation_type:
            case "html" | "accessibility_tree":
                self.text_observation_type = observation_type
                self.image_observation_type = ""
                self.main_observation_type = "text"
            case "image":
                self.image_observation_type = observation_type
                self.text_observation_type = ""  # type: ignore[assignment]
                self.main_observation_type = "image"
            case _:
                raise ValueError(
                    f"Unsupported observation type: {observation_type}"
                )

        self.observation_handler = ObservationHandler(
            self.main_observation_type,
            self.text_observation_type,
            self.image_observation_type,
            self.current_viewport_only,
            self.viewport_size,
        )

        self.observation_space = (
            self.observation_handler.get_observation_space()
        )

    @beartype
    def setup(self) -> None:
        if self.save_trace_enabled:
            self.context.tracing.start(screenshots=True, snapshots=True)

    def get_page_client(self, page: Page) -> CDPSession:
        return page.client  # type: ignore

    async def _get_obs(self) -> dict[str, Observation]:
        obs = await self.observation_handler.get_observation(
            self.page, self.get_page_client(self.page)
        )
        return obs

    def _get_obs_metadata(self) -> dict[str, ObservationMetadata]:
        metadata = self.observation_handler.get_observation_metadata()
        return metadata

    @beartype
    async def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        Reset the environment.
        :param options: options for the environment. The current supported options are:
            - "storage_state": the storage state of the browser. It is a file path to a json file.
        """
        super().reset(seed=seed, options=options)
        if self.reset_finished and self.context_manager:
            self.context_manager.__exit__()

        if options is not None and "config_file" in options:
            config_file = Path(options["config_file"])
            if config_file.exists():
                self.setup(config_file=config_file)
            else:
                raise ValueError(f"Config file {config_file} does not exist.")
        else:
            self.setup()
        self.reset_finished = True

        if self.sleep_after_execution > 0:
            time.sleep(self.sleep_after_execution)
            
        images = await self.modify_page()

        observation = await self._get_obs()
        observation_metadata = self._get_obs_metadata()
        info = {
            "page": DetachedPage(self.page.url, ""),
            "fail_error": "",
            "observation_metadata": observation_metadata,
            "images": images,
        } 

        return (observation, info)

    def save_trace(self, trace_path: str | Path) -> None:
        if self.save_trace_enabled:
            self.context.tracing.stop(path=trace_path)

    async def close(self) -> None:
        if self.reset_finished and self.context_manager:
            await self.context_manager.__exit__()

    async def step(
        self, action: Action
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        if not self.reset_finished:
            raise RuntimeError("Call reset first before calling step.")

        success = False
        fail_error = ""
        try:
            self.page = await execute_action(
                action,
                self.page,
                self.context,
                self.observation_handler.action_processor,
            )
            success = True
        except Exception as e:
            fail_error = str(e)
            raise e

        # hard sleep TODO[shuyanzh] suboptimal, may need to check network
        if self.sleep_after_execution > 0:
            time.sleep(self.sleep_after_execution)

        images = await self.modify_page()
        
        observation = await self._get_obs()
        observation_metadata = self._get_obs_metadata()

        info = {
            "page": DetachedPage(self.page.url, await self.page.content()),
            "fail_error": fail_error,
            "observation_metadata": observation_metadata,
            "images": images,
        }
        
        msg = (
            observation,
            float(success),  # reward
            False,  # terminated
            False,  # truncated
            info,
        )
        return msg

    async def modify_page(self):
        await self.page.wait_for_timeout(500)
        try:
            await self.page.evaluate(remove_id_script)
        except:
            pass
        
        suffix = getattr(self.global_config, "logname", "")
        if suffix:
            img_bytes = await self.page.screenshot(path=f"output/screenshot-{suffix}.png", full_page=True)
        else:
            img_bytes = await self.page.screenshot(path="output/screenshot_raw.png")
        raw_image = base64.b64encode(img_bytes).decode()
        
        await self.page.evaluate(mix_marker_script)
        await self.page.wait_for_timeout(100)
        
        # get all clickable elements
        start_id = 0
        elem_items, start_id = await self.page.evaluate(get_rect_script, {
            "selector": ".possible-clickable-element",
            "startIndex": start_id
        })
        
        # get ocr items
        ocr_items = []
        # ocr_items = page.evaluate(canva_handler_script)
        # svg_items, _ = page.evaluate(get_rect_script, {"selector": "svg", "startIndex": -1})
        # ocr_items = ocr_items + svg_items
        # ocr_items, start_id = get_canva_images(ocr_items, img_bytes, start_id)
        
        items = elem_items + ocr_items
        
        # mark our own labels and get the images
        items = await self.page.evaluate(label_marker_script, items)
        if suffix:
            img_bytes = await self.page.screenshot(path=f"output/marked-{suffix}.png", full_page=True)
        else:
            img_bytes = await self.page.screenshot(path="output/marked.png")
        marked_image = base64.b64encode(img_bytes).decode()
        
        await self.page.evaluate(remove_label_mark_script)
        
        return {
            "raw_image": raw_image,
            "marked_image": marked_image,
        }