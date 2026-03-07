# =============================================================================
# stats_engine.py — Match statistics engine for Football Vision AI
# =============================================================================

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

_PLAYER_LABELS = {"player", "goalkeeper"}
_MAX_PLAYERS_PER_TEAM = 11
_MAX_REFEREES = 4


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


class MatchStats:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._players: dict[int, dict] = {}
        self._ball_position: tuple[float, float] | None = None
        self._ball_possession: str | None = None
        self._frame_count: int = 0
        self._frame_width: int = 0
        self._frame_height: int = 0

        self._class_counts: Counter = Counter()
        self._confidence_sums: defaultdict = defaultdict(float)
        self._confidence_counts: defaultdict = defaultdict(int)

        self._ball_seen_frames: int = 0
        self._possession_frames_A: int = 0
        self._possession_frames_B: int = 0

        self._frame_team_A_counts: list[int] = []
        self._frame_team_B_counts: list[int] = []
        self._frame_referee_counts: list[int] = []

        # Possession timeline — sampled every N frames for the chart
        self._possession_timeline: list[str | None] = []

        # Pitch coverage — unique cells visited (grid-based)
        self._grid_cols = 20
        self._grid_rows = 15
        self._visited_cells_A: set[tuple[int, int]] = set()
        self._visited_cells_B: set[tuple[int, int]] = set()

        # Attacking third presence — frames where team has players in opponent's third
        self._atk_third_frames_A: int = 0
        self._atk_third_frames_B: int = 0

    def set_frame_size(self, width: int, height: int) -> None:
        self._frame_width = width
        self._frame_height = height

    def update(self, detections: dict) -> None:
        self._frame_count += 1
        dets = detections.get("detections", [])

        ball_center: tuple[float, float] | None = None
        frame_team_a = 0
        frame_team_b = 0
        frame_referees = 0
        frame_possession: str | None = None

        team_a_xs: list[float] = []
        team_b_xs: list[float] = []

        w = self._frame_width or 1920
        h = self._frame_height or 1080

        for det in dets:
            label = det.get("label", "")
            pid   = det.get("id")
            team  = det.get("team")
            bbox  = det.get("bbox", [0, 0, 0, 0])
            conf  = det.get("confidence", 0)

            center = _bbox_center(bbox)

            self._class_counts[label] += 1
            self._confidence_sums[label] += conf
            self._confidence_counts[label] += 1

            if label == "ball":
                ball_center = center
                self._ball_position = center
                self._ball_seen_frames += 1
                continue

            if label == "referee":
                frame_referees += 1
                continue

            if label in _PLAYER_LABELS:
                if team == "A":
                    frame_team_a += 1
                    team_a_xs.append(center[0])
                elif team == "B":
                    frame_team_b += 1
                    team_b_xs.append(center[0])

                # Grid cell for pitch coverage
                if team in ("A", "B"):
                    gc = min(int(center[0] / w * self._grid_cols), self._grid_cols - 1)
                    gr = min(int(center[1] / h * self._grid_rows), self._grid_rows - 1)
                    if team == "A":
                        self._visited_cells_A.add((gr, gc))
                    else:
                        self._visited_cells_B.add((gr, gc))

                if pid is not None:
                    if pid in self._players:
                        prev = tuple(self._players[pid]["last_position"])
                        delta = _euclidean(prev, center)
                        if delta < 300:
                            self._players[pid]["distance"] += delta
                        self._players[pid]["last_position"] = list(center)
                        if team and self._players[pid]["team"] is None:
                            self._players[pid]["team"] = team
                    else:
                        self._players[pid] = {
                            "id": pid, "team": team,
                            "distance": 0.0, "last_position": list(center),
                        }

        self._frame_team_A_counts.append(frame_team_a)
        self._frame_team_B_counts.append(frame_team_b)
        self._frame_referee_counts.append(frame_referees)

        # Attacking third: right third = opponent's for A, left third = opponent's for B
        third_boundary = w / 3.0
        if any(x > 2 * third_boundary for x in team_a_xs):
            self._atk_third_frames_A += 1
        if any(x < third_boundary for x in team_b_xs):
            self._atk_third_frames_B += 1

        # Ball possession
        if ball_center is not None:
            nearest_team = self._nearest_player_team(ball_center, dets)
            if nearest_team:
                self._ball_possession = f"Team {nearest_team}"
                frame_possession = nearest_team
                if nearest_team == "A":
                    self._possession_frames_A += 1
                else:
                    self._possession_frames_B += 1

        self._possession_timeline.append(frame_possession)

    def get_stats(self) -> dict:
        avg_conf = {}
        for label in self._confidence_counts:
            avg_conf[label] = round(
                self._confidence_sums[label] / self._confidence_counts[label], 3
            )

        total_poss = self._possession_frames_A + self._possession_frames_B
        poss_pct_a = round(self._possession_frames_A / total_poss * 100, 1) if total_poss > 0 else 0
        poss_pct_b = round(self._possession_frames_B / total_poss * 100, 1) if total_poss > 0 else 0

        ball_visibility = (
            round(self._ball_seen_frames / self._frame_count * 100, 1)
            if self._frame_count > 0 else 0
        )

        median_a = int(statistics.median(self._frame_team_A_counts)) if self._frame_team_A_counts else 0
        median_b = int(statistics.median(self._frame_team_B_counts)) if self._frame_team_B_counts else 0
        median_r = int(statistics.median(self._frame_referee_counts)) if self._frame_referee_counts else 0

        total_cells = self._grid_cols * self._grid_rows
        coverage_a = round(len(self._visited_cells_A) / total_cells * 100, 1)
        coverage_b = round(len(self._visited_cells_B) / total_cells * 100, 1)

        fc = max(1, self._frame_count)
        atk_pct_a = round(self._atk_third_frames_A / fc * 100, 1)
        atk_pct_b = round(self._atk_third_frames_B / fc * 100, 1)

        # Possession timeline — downsample to ~20 data points for the chart
        timeline = self._build_possession_timeline()

        return {
            "frames_analysed": self._frame_count,
            "players_team_A": min(median_a, _MAX_PLAYERS_PER_TEAM),
            "players_team_B": min(median_b, _MAX_PLAYERS_PER_TEAM),
            "referees_on_field": min(median_r, _MAX_REFEREES),
            "ball_possession": self._ball_possession,
            "possession_pct": {"A": poss_pct_a, "B": poss_pct_b},
            "ball_visibility_pct": ball_visibility,
            "class_counts": dict(self._class_counts),
            "avg_confidence": avg_conf,
            "pitch_coverage_pct": {"A": coverage_a, "B": coverage_b},
            "attacking_third_pct": {"A": atk_pct_a, "B": atk_pct_b},
            "possession_timeline": timeline,
        }

    def _build_possession_timeline(self) -> list[dict]:
        """Downsample possession to ~20 buckets for a timeline chart."""
        n = len(self._possession_timeline)
        if n == 0:
            return []
        bucket_count = min(20, n)
        bucket_size = max(1, n // bucket_count)
        result = []
        for i in range(0, n, bucket_size):
            chunk = self._possession_timeline[i:i + bucket_size]
            a_ct = sum(1 for x in chunk if x == "A")
            b_ct = sum(1 for x in chunk if x == "B")
            total = a_ct + b_ct
            result.append({
                "bucket": len(result),
                "pct_A": round(a_ct / total * 100) if total > 0 else 50,
                "pct_B": round(b_ct / total * 100) if total > 0 else 50,
            })
        return result

    def _nearest_player_team(
        self, ball_center: tuple[float, float], dets: list[dict],
    ) -> str | None:
        min_dist = float("inf")
        best_team: str | None = None
        for det in dets:
            if det.get("label") not in _PLAYER_LABELS:
                continue
            if not det.get("team"):
                continue
            center = _bbox_center(det["bbox"])
            dist = _euclidean(ball_center, center)
            if dist < min_dist:
                min_dist = dist
                best_team = det["team"]
        return best_team
