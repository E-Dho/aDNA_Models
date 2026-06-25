#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import math
from pathlib import Path

import imageio.v2 as imageio
from playwright.async_api import async_playwright


async def render_rotation(
    html_path: Path,
    output_path: Path,
    *,
    frames: int,
    width: int,
    height: int,
    radius: float,
    z_eye: float,
    settle_ms: int,
    step_wait_ms: int,
    fps: int,
    browser_channel: str | None,
    executable_path: str | None,
    headless: bool,
    hide_legend: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    captured_frames = []

    async with async_playwright() as playwright:
        launch_kwargs = {
            "headless": headless,
            "args": [
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-first-run",
                "--disable-background-networking",
            ],
        }
        if browser_channel:
            launch_kwargs["channel"] = browser_channel
        if executable_path:
            launch_kwargs["executable_path"] = executable_path
        browser = await playwright.chromium.launch(**launch_kwargs)
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.goto(html_path.resolve().as_uri())
        await page.wait_for_selector(".plotly-graph-div", timeout=15000)
        await page.wait_for_timeout(settle_ms)
        if hide_legend:
            await page.evaluate(
                """() => {
                    const gd = document.querySelector('.plotly-graph-div');
                    if (!gd) throw new Error('No .plotly-graph-div found in HTML');
                    Plotly.relayout(gd, {'showlegend': false});
                }"""
            )
            await page.wait_for_timeout(100)

        for i in range(frames):
            theta = 2.0 * math.pi * i / frames
            eye = {
                "x": radius * math.cos(theta),
                "y": radius * math.sin(theta),
                "z": z_eye,
            }

            await page.evaluate(
                """eye => {
                    const gd = document.querySelector('.plotly-graph-div');
                    if (!gd) throw new Error('No .plotly-graph-div found in HTML');
                    Plotly.relayout(gd, {'scene.camera.eye': eye});
                }""",
                eye,
            )
            await page.wait_for_timeout(step_wait_ms)
            png_bytes = await page.screenshot(type="png")
            captured_frames.append(imageio.imread(png_bytes))

        await browser.close()

    imageio.mimsave(output_path, captured_frames, fps=fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a rotating MP4 from a Plotly latent-space HTML plot.")
    parser.add_argument("--html", required=True, help="Input Plotly HTML file")
    parser.add_argument("--output", required=True, help="Output .mp4 path")
    parser.add_argument("--frames", type=int, default=72, help="Number of frames in the rotation")
    parser.add_argument("--width", type=int, default=900, help="Viewport width in pixels")
    parser.add_argument("--height", type=int, default=700, help="Viewport height in pixels")
    parser.add_argument("--radius", type=float, default=1.8, help="Camera orbit radius")
    parser.add_argument("--z_eye", type=float, default=0.9, help="Camera z value")
    parser.add_argument("--settle_ms", type=int, default=1500, help="Initial wait time after page load")
    parser.add_argument("--step_wait_ms", type=int, default=40, help="Wait time after each camera move")
    parser.add_argument("--fps", type=int, default=20, help="Output video frames per second")
    parser.add_argument("--browser_channel", default=None, help="Optional Playwright browser channel, e.g. chrome")
    parser.add_argument(
        "--executable_path",
        default=None,
        help="Optional explicit browser executable path, e.g. /Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run with a visible browser window instead of headless mode",
    )
    parser.add_argument(
        "--hide_legend",
        action="store_true",
        help="Hide the Plotly legend during video capture so it does not cover the plot",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(
        render_rotation(
            html_path=Path(args.html),
            output_path=Path(args.output),
            frames=args.frames,
            width=args.width,
            height=args.height,
            radius=args.radius,
            z_eye=args.z_eye,
            settle_ms=args.settle_ms,
            step_wait_ms=args.step_wait_ms,
            fps=args.fps,
            browser_channel=args.browser_channel,
            executable_path=args.executable_path,
            headless=not args.headed,
            hide_legend=args.hide_legend,
        )
    )


if __name__ == "__main__":
    main()
