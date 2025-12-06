"""自定义带彩色 Spinner 旋转动画的 RichProgressBar"""

from typing import Any, Optional

from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBar,
    RichProgressBarTheme,
    CustomBarColumn,
    BatchesProcessedColumn,
    CustomTimeColumn,
    ProcessingSpeedColumn,
)
from rich.progress import TextColumn, ProgressColumn, Task
from rich.text import Text


class ColorfulSpinnerColumn(ProgressColumn):
    """每帧不同颜色的 Spinner 列"""

    FRAMES = "^}/:>*┐~>-<╰┌*<:\\{^"
    COLORS = ["#AE9389", "#43BAC9", "#BF5598", "#58718A", "#63C66E", "#D5C6C9"]

    def __init__(self, speed: float = 10.0, finished_text: str = " "):
        super().__init__()
        self.speed = speed
        self.finished_text = finished_text

    def render(self, task: Task) -> Text:
        if task.finished:
            return Text(self.finished_text)

        elapsed = task.elapsed or 0
        frame_index = int(elapsed * self.speed) % len(self.FRAMES)
        char = self.FRAMES[frame_index]
        color = self.COLORS[frame_index % len(self.COLORS)]

        return Text(char, style=color)


class SpinnerRichProgressBar(RichProgressBar):
    """带彩色 Spinner 旋转动画的进度条

    在原有 RichProgressBar 的基础上，在最前面添加一个彩色 spinner 动画列。
    18 个字符按 6 种颜色循环变换。
    """

    def __init__(
        self,
        spinner_speed: float = 10.0,
        refresh_rate: int = 1,
        leave: bool = False,
        theme: RichProgressBarTheme = RichProgressBarTheme(),
        console_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            refresh_rate=refresh_rate,
            leave=leave,
            theme=theme,
            console_kwargs=console_kwargs,
        )
        self.spinner_speed = spinner_speed

    def configure_columns(self, trainer) -> list:
        """覆写此方法，在原有列前添加彩色 SpinnerColumn"""
        return [
            ColorfulSpinnerColumn(speed=self.spinner_speed),
            TextColumn("[progress.description]{task.description}"),
            CustomBarColumn(
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            CustomTimeColumn(style=self.theme.time),
            ProcessingSpeedColumn(style=self.theme.processing_speed),
        ]
