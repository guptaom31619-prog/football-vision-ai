# =============================================================================
# heatmap.py — Team-level heatmaps rendered on a proper soccer pitch
#
# Generates exactly 3 heatmaps: Team A, Team B, All Players.
# Each is drawn on a realistic football pitch background with field lines,
# center circle, penalty areas, goal areas, and corner arcs.
# =============================================================================

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np


_PLAYER_LABELS = {"player", "goalkeeper"}
_BLUR_KERNEL = (201, 201)
_BLUR_SIGMA  = 80
_OUTPUT_DIR  = Path(__file__).parent / "heatmaps"


def _draw_pitch(ax: plt.Axes, w: int, h: int) -> None:
    """Draw a regulation football pitch scaled to pixel dimensions w×h."""
    lc = "#ffffff"
    lw = 1.2
    al = 0.45

    # Outer boundary
    ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color=lc, lw=lw + 0.5, alpha=al)

    # Halfway line
    ax.plot([w / 2, w / 2], [0, h], color=lc, lw=lw, alpha=al)

    # Center circle (radius ~9.15m on 105m pitch → ~8.7% of width)
    r = w * 0.087
    circle = patches.Circle((w / 2, h / 2), r, fill=False, ec=lc, lw=lw, alpha=al)
    ax.add_patch(circle)

    # Center spot
    ax.plot(w / 2, h / 2, "o", color=lc, ms=3, alpha=al)

    # Penalty areas (16.5m deep on 105m → ~15.7%, 40.3m wide on 68m → ~59.3%)
    pa_w = w * 0.157
    pa_h = h * 0.593
    pa_y = (h - pa_h) / 2

    # Left penalty area
    ax.plot([0, pa_w, pa_w, 0], [pa_y, pa_y, pa_y + pa_h, pa_y + pa_h],
            color=lc, lw=lw, alpha=al)
    # Right penalty area
    ax.plot([w, w - pa_w, w - pa_w, w], [pa_y, pa_y, pa_y + pa_h, pa_y + pa_h],
            color=lc, lw=lw, alpha=al)

    # Goal areas (5.5m deep on 105m → ~5.2%, 18.3m wide on 68m → ~26.9%)
    ga_w = w * 0.052
    ga_h = h * 0.269
    ga_y = (h - ga_h) / 2

    ax.plot([0, ga_w, ga_w, 0], [ga_y, ga_y, ga_y + ga_h, ga_y + ga_h],
            color=lc, lw=lw, alpha=al)
    ax.plot([w, w - ga_w, w - ga_w, w], [ga_y, ga_y, ga_y + ga_h, ga_y + ga_h],
            color=lc, lw=lw, alpha=al)

    # Penalty spots (11m from goal line → ~10.5% of width)
    ps_x = w * 0.105
    ax.plot(ps_x, h / 2, "o", color=lc, ms=2.5, alpha=al)
    ax.plot(w - ps_x, h / 2, "o", color=lc, ms=2.5, alpha=al)

    # Penalty arcs (arc outside penalty area around penalty spot)
    arc_r = w * 0.087
    arc_l = patches.Arc((ps_x, h / 2), arc_r * 2, arc_r * 2,
                         angle=0, theta1=-50, theta2=50,
                         ec=lc, lw=lw, alpha=al)
    arc_r_patch = patches.Arc((w - ps_x, h / 2), arc_r * 2, arc_r * 2,
                               angle=0, theta1=130, theta2=230,
                               ec=lc, lw=lw, alpha=al)
    ax.add_patch(arc_l)
    ax.add_patch(arc_r_patch)

    # Corner arcs
    cr = w * 0.015
    for cx, cy, t1, t2 in [
        (0, 0, 0, 90), (w, 0, 90, 180),
        (w, h, 180, 270), (0, h, 270, 360),
    ]:
        ca = patches.Arc((cx, cy), cr * 2, cr * 2,
                         angle=0, theta1=t1, theta2=t2,
                         ec=lc, lw=lw, alpha=al)
        ax.add_patch(ca)


class HeatmapGenerator:
    """Accumulates player positions and renders team-level pitch heatmaps."""

    def __init__(self) -> None:
        self.player_positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self.player_teams: dict[int, str | None] = {}
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        self.player_positions.clear()
        self.player_teams.clear()

    def update(self, detections: dict) -> None:
        for det in detections.get("detections", []):
            label = det.get("label", "")
            pid   = det.get("id")
            team  = det.get("team")
            bbox  = det.get("bbox", [0, 0, 0, 0])

            if label not in _PLAYER_LABELS or pid is None:
                continue

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            self.player_positions[pid].append((cx, cy))

            if pid not in self.player_teams or self.player_teams[pid] is None:
                self.player_teams[pid] = team

    def generate_all(self, frame_shape: tuple[int, int]) -> list[Path]:
        saved: list[Path] = []
        for team_key, cmap, label in [
            ("A", "plasma",  "Team A"),
            ("B", "inferno", "Team B"),
        ]:
            path = self._build_team_heatmap(team_key, frame_shape, cmap, label)
            if path:
                saved.append(path)

        all_path = self._build_combined_heatmap(frame_shape)
        if all_path:
            saved.append(all_path)
        return saved

    def generate_heatmap(self, player_id: int, frame_shape: tuple[int, int]) -> Path | None:
        return None

    def _collect_positions(self, team: str | None) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for pid, positions in self.player_positions.items():
            if team is not None and self.player_teams.get(pid) != team:
                continue
            pts.extend(positions)
        return pts

    def _count_players(self, team: str | None) -> int:
        """Return capped player count (max 11 per team, max 22 total)."""
        if team is None:
            return min(len(self.player_positions), 22)
        raw = sum(1 for pid in self.player_positions if self.player_teams.get(pid) == team)
        return min(raw, 11)

    def _build_matrix(self, positions: list[tuple[float, float]], h: int, w: int) -> np.ndarray:
        matrix = np.zeros((h, w), dtype=np.float32)
        # Splat each position as a filled circle, not a single pixel.
        # Radius scales with frame size so heat zones are visible.
        radius = max(12, int(min(w, h) * 0.02))
        for cx, cy in positions:
            px = int(np.clip(cx, 0, w - 1))
            py = int(np.clip(cy, 0, h - 1))
            cv2.circle(matrix, (px, py), radius, 1.0, -1)
        blurred = cv2.GaussianBlur(matrix, _BLUR_KERNEL, _BLUR_SIGMA)
        mx = blurred.max()
        if mx > 0:
            blurred /= mx
        return blurred

    def _build_team_heatmap(
        self, team: str, frame_shape: tuple[int, int], cmap: str, label: str,
    ) -> Path | None:
        positions = self._collect_positions(team)
        if not positions:
            return None
        h, w = frame_shape
        matrix = self._build_matrix(positions, h, w)
        n_players = self._count_players(team)
        title = f"{label}  —  {n_players} players  |  {len(positions):,} data points"
        out_path = _OUTPUT_DIR / f"team_{team}.png"
        _render(matrix, title, cmap, out_path, w, h)
        return out_path

    def _build_combined_heatmap(self, frame_shape: tuple[int, int]) -> Path | None:
        positions = self._collect_positions(None)
        if not positions:
            return None
        h, w = frame_shape
        matrix = self._build_matrix(positions, h, w)
        n_players = self._count_players(None)
        title = f"All Players  —  {n_players} tracked  |  {len(positions):,} data points"
        out_path = _OUTPUT_DIR / "all_players.png"
        _render(matrix, title, "inferno", out_path, w, h)
        return out_path


def _render(
    matrix: np.ndarray, title: str, cmap_name: str,
    out_path: Path, w: int, h: int,
) -> None:
    """
    Render heatmap by manually compositing an RGBA heat layer onto a pitch image.
    No matplotlib alpha tricks — pure pixel blending that always works.
    """
    # --- 1. Build pitch image (RGB uint8) ---
    pitch = np.zeros((h, w, 3), dtype=np.uint8)
    stripe_w = w // 12
    for i in range(12):
        c = (30, 82, 30) if i % 2 == 0 else (26, 71, 26)
        pitch[:, i * stripe_w:(i + 1) * stripe_w] = c

    # Draw pitch lines directly with OpenCV (white, thin)
    lc = (255, 255, 255)
    lt = max(1, w // 600)
    # Boundary
    cv2.rectangle(pitch, (0, 0), (w - 1, h - 1), lc, lt)
    # Halfway
    cv2.line(pitch, (w // 2, 0), (w // 2, h), lc, lt)
    # Center circle
    cv2.circle(pitch, (w // 2, h // 2), int(w * 0.087), lc, lt)
    cv2.circle(pitch, (w // 2, h // 2), 4, lc, -1)
    # Penalty areas
    pa_w, pa_h = int(w * 0.157), int(h * 0.593)
    pa_y = (h - pa_h) // 2
    cv2.rectangle(pitch, (0, pa_y), (pa_w, pa_y + pa_h), lc, lt)
    cv2.rectangle(pitch, (w - pa_w, pa_y), (w, pa_y + pa_h), lc, lt)
    # Goal areas
    ga_w, ga_h = int(w * 0.052), int(h * 0.269)
    ga_y = (h - ga_h) // 2
    cv2.rectangle(pitch, (0, ga_y), (ga_w, ga_y + ga_h), lc, lt)
    cv2.rectangle(pitch, (w - ga_w, ga_y), (w, ga_y + ga_h), lc, lt)
    # Penalty spots
    ps_x = int(w * 0.105)
    cv2.circle(pitch, (ps_x, h // 2), 3, lc, -1)
    cv2.circle(pitch, (w - ps_x, h // 2), 3, lc, -1)

    # --- 2. Build heat overlay (RGB) from colormap ---
    boosted = np.power(np.clip(matrix, 0, 1), 0.4)
    cmap = plt.get_cmap(cmap_name)
    heat_rgba = cmap(boosted)  # (h, w, 4) float 0-1
    heat_rgb = (heat_rgba[:, :, :3] * 255).astype(np.uint8)

    # --- 3. Alpha blend: pitch * (1 - a) + heat * a ---
    # Alpha ramps from 0 (no data) to 0.85 (hot spots)
    a = np.clip(boosted * 1.5, 0, 0.85)
    a[boosted < 0.02] = 0.0  # fully transparent where no data
    a3 = np.stack([a, a, a], axis=-1)  # broadcast to 3 channels

    composited = (pitch.astype(np.float32) * (1 - a3) + heat_rgb.astype(np.float32) * a3)
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    # --- 4. Render with matplotlib (just the composited image + title/colorbar) ---
    dpi = 100
    fig_w = max(7, w / dpi)
    fig_h = max(5, h / dpi)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#0f1923")

    ax.imshow(composited, extent=[0, w, h, 0])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, shrink=0.7)
    cbar.set_label("Presence", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="#0f1923")
    plt.close(fig)
